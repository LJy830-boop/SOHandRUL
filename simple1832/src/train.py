import os
import sys
from pathlib import Path
import glob

def _auto_set_data_env():
    d_env = os.environ.get('BATTERY_XLSX_DIR', '').strip()
    f_env = os.environ.get('BATTERY_XLSX_PATH', '').strip()
    if d_env or f_env:
        return
    cands = []
    cwd = Path.cwd()
    cands.append(cwd / 'data')
    cands.append(cwd.parent / 'data')
    cands.append(Path(__file__).resolve().parent.parent / 'data')
    if os.name != 'nt':
        cands.append(Path('/data/home/scutljy/SOH-V4/data'))
    picked = None
    for p in cands:
        try:
            if p.exists():
                files = list(p.rglob('*.csv')) + list(p.rglob('*.xlsx'))
                if files:
                    picked = p
                    break
        except Exception:
            pass
    if picked is not None:
        os.environ['BATTERY_XLSX_DIR'] = str(picked)

_auto_set_data_env()

# ========== 强制开启数据平滑，确保训练和预测一致 ==========
os.environ['SMOOTH_RAW_CAPACITY'] = '1'  # 平滑原始容量数据
os.environ['SOH_SMOOTH_OUTLIERS'] = '1'  # 平滑SOH异常值
os.environ['RUL_SMOOTH'] = '1'  # 平滑RUL标签

import glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, GroupKFold, LeaveOneGroupOut, cross_val_predict, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from feature_selectors import CombinedFeatureSelector, CorrelationFilter as FSCorrelationFilter
from models import choose_base_models_by_data, build_stacking, BlendingRegressor, RFPlusANNRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import hashlib
from typing import Optional
from build_dataset import build_dataset_from_excel
import joblib
import json

CorrelationFilter = FSCorrelationFilter

# 新增：稳定、全面的特征预处理器
class StableFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        missing_ratio_threshold: float = 0.95,
        clip_quantiles: tuple = (0.01, 0.99),
        variance_threshold: float = 1e-12,
        corr_threshold: float = 0.85,
        corr_method: str = 'pearson',
        use_vif: bool = False,
        vif_threshold: float = 10.0,
        apply_power_transform: bool = False,
        scale_mode: str = 'robust'
    ):
        # 保留用户传入值原样，确保 sklearn 可克隆
        self.missing_ratio_threshold = missing_ratio_threshold
        self.clip_quantiles = clip_quantiles
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.corr_method = corr_method
        self.use_vif = use_vif
        self.vif_threshold = vif_threshold
        self.apply_power_transform = apply_power_transform
        self.scale_mode = scale_mode

    def _to_dataframe(self, X):
        import numpy as np
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.copy()
        arr = X if isinstance(X, np.ndarray) else np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f'f_{i}' for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    def _to_numeric_inplace(self, df):
        import pandas as pd
        import numpy as np
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                try:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                except Exception:
                    df[c] = pd.Series(np.nan, index=df.index)

    def fit(self, X, y=None):
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer

        # 在 fit 内做规范化处理，避免 clone 误判
        missing_ratio_threshold = float(self.missing_ratio_threshold)
        clip_low, clip_high = float(self.clip_quantiles[0]), float(self.clip_quantiles[1])
        variance_threshold = float(self.variance_threshold)
        corr_threshold = float(self.corr_threshold)
        corr_method = str(self.corr_method).lower()
        use_vif = bool(self.use_vif)
        vif_threshold = float(self.vif_threshold)
        scale_mode = str(self.scale_mode).lower()

        df = self._to_dataframe(X)
        self.feature_names_in_ = list(df.columns)

        # 数值化
        self._to_numeric_inplace(df)

        # 去重列
        try:
            dup_mask = df.T.duplicated()
            removed_exact_duplicates = [col for col, dup in zip(df.columns, dup_mask) if dup]
            df = df.loc[:, ~dup_mask]
        except Exception:
            removed_exact_duplicates = []

        # 高缺失剔除
        miss_ratio = df.isna().mean()
        keep_miss = miss_ratio <= missing_ratio_threshold
        removed_high_missing = [col for col, keep in zip(df.columns, keep_miss) if not keep]
        df = df.loc[:, keep_miss.values]

        # 中位数填补（先算，用于 transform 复用）
        med = df.median()
        self.medians_ = {c: float(med.get(c, 0.0)) for c in df.columns}
        df = df.fillna(med)

        # 分位裁剪（winsorize）
        q_low = df.quantile(clip_low)
        q_high = df.quantile(clip_high)
        self.q_low_ = {c: float(q_low.get(c, df[c].min())) for c in df.columns}
        self.q_high_ = {c: float(q_high.get(c, df[c].max())) for c in df.columns}
        df = df.clip(lower=q_low, upper=q_high, axis=1)

        # 低方差剔除
        var = df.var(ddof=0)
        keep_var = var > variance_threshold
        removed_low_variance = [col for col, keep in zip(df.columns, keep_var) if not keep]
        df = df.loc[:, keep_var.values]

        # 相关性去冗余（贪心）
        corr_df = df.corr(method=('spearman' if corr_method == 'spearman' else 'pearson'))
        corr = np.nan_to_num(corr_df.values, nan=0.0)
        n = corr.shape[0]
        to_drop = set()
        keep_idx = []
        for i in range(n):
            if i in to_drop:
                continue
            keep_idx.append(i)
            for j in range(i + 1, n):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) >= corr_threshold:
                    to_drop.add(j)
        removed_high_corr = [df.columns[j] for j in range(n) if j in to_drop]
        df = df.iloc[:, keep_idx]

        # 可选：VIF 剔除（迭代移除最大者）
        removed_high_vif = []
        if use_vif and df.shape[1] >= 6:
            try:
                import numpy as np
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                # 迭代上限防止死循环
                for _ in range(min(50, df.shape[1])):
                    X_mat = np.asarray(df.values, dtype=float)
                    vifs = [variance_inflation_factor(X_mat, i) for i in range(X_mat.shape[1])]
                    vifs = np.asarray(vifs, dtype=float)
                    if not np.isfinite(vifs).all():
                        break
                    if vifs.max() <= vif_threshold or X_mat.shape[1] <= 5:
                        break
                    drop_idx = int(vifs.argmax())
                    removed_high_vif.append(df.columns[drop_idx])
                    df = df.drop(columns=[df.columns[drop_idx]])
            except Exception:
                # statsmodels 未安装或失败则跳过
                pass

        # 可选：幂变换（Yeo-Johnson），减轻偏态（不标准化）
        self._power_ = None
        if self.apply_power_transform and df.shape[1] > 0:
            try:
                self._power_ = PowerTransformer(method='yeo-johnson', standardize=False)
                _arr = np.asarray(self._power_.fit_transform(df))
                if _arr.ndim == 1:
                    _arr = _arr.reshape(-1, 1)
                _cols = list(df.columns)
                if _arr.shape[1] != len(_cols):
                    _cols = [f'f_{i}' for i in range(_arr.shape[1])]
                df = pd.DataFrame(_arr, index=df.index, columns=_cols)
            except Exception:
                self._power_ = None

        # 缩放（稳健优先）
        scaler = RobustScaler() if scale_mode == 'robust' else StandardScaler()
        self._scaler_ = scaler.fit(df.values)
        scaled = self._scaler_.transform(df.values)
        _arr = np.asarray(scaled)
        if _arr.ndim == 1:
            _arr = _arr.reshape(-1, 1)
        _cols = list(df.columns)
        if _arr.shape[1] != len(_cols):
            _cols = [f'f_{i}' for i in range(_arr.shape[1])]
        df_scaled = pd.DataFrame(_arr, index=df.index, columns=_cols)

        # 记录
        self.keep_columns_ = list(df.columns)
        self.feature_names_out_ = list(df_scaled.columns)
        self.report_ = {
            'removed_exact_duplicates': removed_exact_duplicates,
            'removed_high_missing': removed_high_missing,
            'removed_low_variance': removed_low_variance,
            'removed_high_corr': removed_high_corr,
            'removed_high_vif': removed_high_vif
        }
        return self

    def transform(self, X):
        import numpy as np
        import pandas as pd
        df = self._to_dataframe(X)
        # 数值化
        self._to_numeric_inplace(df)
        # 对齐列集合
        df = df.reindex(columns=self.keep_columns_, fill_value=np.nan)
        # 填补缺失（复用中位数）
        fill_vals = {c: self.medians_.get(c, 0.0) for c in df.columns}
        df = df.fillna(value=fill_vals)
        # 分位裁剪（复用阈值）
        for c in df.columns:
            low = self.q_low_.get(c, None)
            high = self.q_high_.get(c, None)
            if low is not None or high is not None:
                df[c] = df[c].clip(lower=low, upper=high)
        # 幂变换（如启用）
        if getattr(self, '_power_', None) is not None:
            try:
                _arr = np.asarray(self._power_.transform(df))
                if _arr.ndim == 1:
                    _arr = _arr.reshape(-1, 1)
                _cols = list(df.columns)
                if _arr.shape[1] != len(_cols):
                    _cols = [f'f_{i}' for i in range(_arr.shape[1])]
                df = pd.DataFrame(_arr, index=df.index, columns=_cols)
            except Exception:
                pass
        # 缩放
        arr = self._scaler_.transform(df.values)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = list(self.feature_names_out_)
        if arr.shape[1] != len(cols):
            cols = [f'f_{i}' for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

# 检查GPU支持
try:
    import torch
    if torch.cuda.is_available():
        print(f"PyTorch GPU可用，设备数量: {torch.cuda.device_count()}")
        device = torch.device('cuda')
        # 设置使用所有可用GPU
        visible_devices = ','.join(str(i) for i in range(torch.cuda.device_count()))
        if visible_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    else:
        print("PyTorch GPU不可用，使用CPU")
        device = torch.device('cpu')
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch未安装")
    TORCH_AVAILABLE = False
    device = None

# 检查XGBoost GPU支持
try:
    import xgboost as xgb
    print("XGBoost可用，将使用GPU加速")
    XGB_GPU_AVAILABLE = True
except ImportError:
    print("XGBoost不可用")
    XGB_GPU_AVAILABLE = False

# Optional YAML support
try:
    import yaml
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False

# 添加timeout函数定义
# 彻底移除所有超时相关代码，避免任何导入错误
# 不再使用任何超时机制
def timeout(seconds):
    """简化的超时处理函数 - 兼容所有Python版本"""
    class DummyContext:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
    return DummyContext()

def save_config(cfg: dict, art_dir: str):
    os.makedirs(art_dir, exist_ok=True)
    out_yaml = os.path.join(art_dir, 'config.yaml')
    out_json = os.path.join(art_dir, 'config.json')
    if _HAS_YAML and yaml is not None:
        try:
            with open(out_yaml, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            return
        except Exception:
            pass
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# 修改：评估函数支持分组/留一电池交叉验证
def evaluate_model(name, model, X, y, cv_splits: int = 5, cv_strategy='kfold', selector=None, groups=None, enforce_cv: bool = True):
    global np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    print(f"\n开始评估模型: {name}")
    print(f"数据形状: {X.shape}, 目标形状: {y.shape}")

    # 移除全量预清洗，避免泄漏；改为仅依赖管道内的分折预处理

    # 根据数据规模调整交叉验证策略
    sample_count = X.shape[0]
    n_features = X.shape[1]
    if not enforce_cv:
        if sample_count < 200:
            cv_splits = min(3, cv_splits)
            print(f"小样本数据，调整为 {cv_splits} 折交叉验证")
        elif sample_count < 1000:
            cv_splits = min(4, cv_splits)
            print(f"中等样本数据，调整为 {cv_splits} 折交叉验证")
        if n_features > 500:
            cv_splits = 3
            print("高维数据，调整为 3 折交叉验证")
    if groups is not None and cv_strategy in ('kfold', 'group', 'groupkfold', 'group_kfold'):
        n_groups = int(len(np.unique(groups)))
        if cv_strategy in ('groupkfold', 'group_kfold', 'group'):
            if n_groups < 2:
                print("警告：分组数量不足（<2），切换为KFold")
                cv_strategy = 'kfold'
            elif cv_splits > n_groups:
                print(f"注意：n_splits({cv_splits}) > n_groups({n_groups})，自动调整为 {n_groups}")
                cv_splits = n_groups

    sel = clone(selector) if selector is not None else None

    if cv_strategy == 'timeseries':
        cv_obj = TimeSeriesSplit(n_splits=cv_splits)
    elif cv_strategy in ('groupkfold','group_kfold','group'):
        cv_obj = GroupKFold(n_splits=cv_splits)
    elif cv_strategy in ('logo','leaveonegroupout','leave_one_group_out'):
        cv_obj = LeaveOneGroupOut()
    elif cv_strategy in ('group_holdout','groupholdout','holdout_groups'):
        cv_obj = None
    else:
        cv_obj = cv_splits

    steps = [
        ('preproc', StableFeaturePreprocessor(
            missing_ratio_threshold=0.95,
            clip_quantiles=(0.01, 0.99),
            variance_threshold=1e-12,
            corr_threshold=0.95,   # 提高去冗余阈值，保留更多非冗余特征
            corr_method='pearson',
            use_vif=False,
            vif_threshold=10.0,
            apply_power_transform=False,
            scale_mode='robust'
        ))
    ]
    if sel is not None:
        steps.append(('select', sel))
    model_clone = clone(model)
    try:
        params = getattr(model_clone, 'get_params', lambda **_: {})()
        if 'cv' in params:
            model_clone.set_params(cv=cv_splits)
    except Exception:
        pass

    # 统一采用更健壮的对比函数绘制容量曲线（自动识别列/支持keys对齐）
    try:
        _ = plot_capacity_compare_ah(xlsx_path, data_mode, None, art_dir, label_lang, keys_df=None, y_true_used=None, pred_label=name)
    except Exception:
        pass
    steps.append(('model', model_clone))
    pipeline = Pipeline(steps)
    print(f"Pipeline构建完成，包含步骤: {[step[0] for step in steps]}")
    print(f"开始交叉验证，使用 {cv_strategy} 策略，{cv_splits} 折")

    try:
        # 特殊：RUL任务支持按电池固定训练/预测窗（前40%/后60%），若已提供预定义划分则直接使用
        if (str(target_type).lower() == 'rul') and (PREDEFINED_SPLIT is not None):
            train_idx, test_idx = PREDEFINED_SPLIT
            print(f"使用预定义按电池时序划分：训练样本={len(train_idx)}，预测样本={len(test_idx)}")
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            rul_mode = os.environ.get('RUL_MODE', '').strip().lower()
            if rul_mode == 'two_stage':
                print('RUL_MODE=two_stage：采用保持率→EOL→RUL两段法')
                # 1) 使用容量保持率(%)作为目标重新构建数据集
                try:
                    if data_mode == 'dir' and os.path.isdir(xlsx_dir):
                        X_ret, y_ret, groups_ret, keys_ret = build_dataset_from_directory(xlsx_dir, target_type='capacity_retention', capacity_threshold_pct=threshold_pct, return_keys=True)
                    else:
                        X_ret, y_ret, keys_ret = build_dataset_from_excel(xlsx_path, target_type='capacity_retention', capacity_threshold_pct=threshold_pct, return_keys=True)
                        groups_ret = None
                except Exception as e:
                    print(f"两段法：保持率数据构建失败，回退直接RUL: {e}")
                    pipeline.fit(X_train, y[train_idx])
                    y_pred = pipeline.predict(X_test)
                else:
                    # 2) 对齐索引（假设构建顺序一致；若失败则使用位置对齐）
                    try:
                        m_train = np.zeros(len(X_ret), dtype=bool); m_train[train_idx] = True
                        m_test = np.zeros(len(X_ret), dtype=bool); m_test[test_idx] = True
                        X_ret_train = X_ret.iloc[train_idx] if hasattr(X_ret, 'iloc') else X_ret[train_idx]
                        X_ret_test = X_ret.iloc[test_idx] if hasattr(X_ret, 'iloc') else X_ret[test_idx]
                        y_ret_train = y_ret[train_idx]
                    except Exception:
                        X_ret_train = X_ret
                        X_ret_test = X_ret
                        y_ret_train = y_ret
                    # 3) 使用稳健管道预测保持率(%)
                    from sklearn.pipeline import Pipeline as _Pipe
                    prep = _Pipe([
                        ('preproc', StableFeaturePreprocessor(
                            missing_ratio_threshold=0.95,
                            clip_quantiles=(0.01, 0.99),
                            variance_threshold=1e-12,
                            corr_threshold=0.90,
                            corr_method='pearson',
                            use_vif=False,
                            scale_mode='robust'
                        )),
                        ('model', base_models.get('xgb', base_models.get('rf')) if 'base_models' in globals() else get_base_models(42, cv_splits).get('xgb'))
                    ])
                    try:
                        prep.fit(X_ret_train, y_ret_train)
                        y_ret_pred = prep.predict(X_ret_test)
                    except Exception:
                        from sklearn.linear_model import Ridge
                        fallback = _Pipe([
                            ('preproc', StableFeaturePreprocessor()),
                            ('model', Ridge())
                        ])
                        fallback.fit(X_ret_train, y_ret_train)
                        y_ret_pred = fallback.predict(X_ret_test)
                    # 4) 转换为预测RUL（按电池）
                    # 需要 cycles 与分组ID 来计算每电池的EOL与RUL
                    try:
                        if isinstance(keys_ret, pd.DataFrame) and 'cycle' in keys_ret.columns:
                            cycles_all = np.asarray(keys_ret['cycle'].values, dtype=float)
                        else:
                            cycles_all = np.arange(len(y_ret))
                        ids_all = None
                        if isinstance(keys_ret, pd.DataFrame) and '条码' in keys_ret.columns:
                            ids_all = keys_ret['条码'].astype(str).values
                        elif groups_ret is not None:
                            ids_all = np.asarray(groups_ret).astype(str)
                        else:
                            ids_all = np.array(['group'] * len(y_ret), dtype=object)
                        cycles_test = cycles_all[test_idx]
                        ids_test = ids_all[test_idx]
                        # 基准容量（原始CSV首圈容量）
                        base_by_id = {}
                        data_dir = os.environ.get('BATTERY_XLSX_DIR', '').strip()
                        for uid in pd.unique(ids_test):
                            base_by_id[str(uid)] = None
                            raw_path = os.path.join(data_dir, str(uid)) if (data_dir and isinstance(uid, (str, bytes))) else ''
                            try:
                                if raw_path and os.path.isfile(raw_path):
                                    from build_dataset import read_battery_csv
                                    s1_raw, _, _ = read_battery_csv(raw_path)
                                    if ('cycle' in s1_raw.columns) and ('discharge_capacity_ah' in s1_raw.columns):
                                        dfr = s1_raw[['cycle','discharge_capacity_ah']].dropna().groupby('cycle', sort=True)['discharge_capacity_ah'].mean()
                                        if dfr.shape[0]:
                                            base_by_id[str(uid)] = float(dfr.iloc[0])
                            except Exception:
                                pass
                        # 转换保持率到容量Ah
                        cap_pred_ah = []
                        for r, cyc, gid in zip(np.asarray(y_ret_pred, dtype=float), cycles_test, ids_test):
                            b = base_by_id.get(str(gid), None)
                            if b is None or not np.isfinite(b):
                                b = 1.0
                            cap_pred_ah.append(max(0.0, b * max(0.0, min(100.0, r)) / 100.0))
                        cap_pred_ah = np.asarray(cap_pred_ah, dtype=float)
                        # 计算每电池预测EOL与RUL
                        thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
                        y_pred_rul = np.empty_like(y_ret_pred, dtype=float)
                        y_pred_rul[:] = np.nan
                        for uid in pd.unique(ids_test):
                            idx_u = np.where(ids_test == uid)[0]
                            if idx_u.size < 2:
                                continue
                            # 阈值容量
                            b = base_by_id.get(str(uid), 1.0)
                            eol_ah = b * thr_pct / 100.0
                            cyc_u = cycles_test[idx_u]
                            cap_u = cap_pred_ah[idx_u]
                            pos = np.where(cap_u < eol_ah)[0]
                            if pos.size > 0:
                                eol_cycle = float(cyc_u[pos[0]])
                            else:
                                eol_cycle = float(np.max(cyc_u))
                            y_pred_rul[idx_u] = np.clip(eol_cycle - cyc_u, 0.0, np.max(y[test_idx]))
                        y_pred = y_pred_rul
                    except Exception as e:
                        print(f"两段法：RUL转换失败，回退直接RUL: {e}")
                        pipeline.fit(X_train, y[train_idx])
                        y_pred = pipeline.predict(X_test)
            else:
                pipeline.fit(X_train, y[train_idx])
                y_pred = pipeline.predict(X_test)
            # 重置当前作用域的 X/y 为预测子集，保证后续 mask 与维度一致
            X = X_test
            y = y[test_idx]
            indices = np.asarray(test_idx, dtype=int)
        elif (cv_strategy in ('group_holdout','groupholdout','holdout_groups')) and (groups is not None):
            uniq = np.unique(np.asarray(groups).astype(object))
            rng = np.random.RandomState(int(os.environ.get('GROUP_HOLDOUT_SEED','42')))
            order = rng.permutation(len(uniq))
            ratio = float(os.environ.get('GROUP_HOLDOUT_RATIO','0.8'))
            if ratio <= 0.0 or ratio >= 1.0:
                ratio = 0.8
            k = max(1, int(np.floor(len(uniq) * ratio)))
            train_groups = uniq[order[:k]]
            test_groups = uniq[order[k:]]
            m_train = np.isin(np.asarray(groups).astype(object), train_groups)
            m_test = np.isin(np.asarray(groups).astype(object), test_groups)
            train_idx = np.where(m_train)[0]
            test_idx = np.where(m_test)[0]
            print(f"分组留出：训练组={len(train_groups)}，测试组={len(test_groups)}；训练样本={len(train_idx)}，测试样本={len(test_idx)}")
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            pipeline.fit(X_train, y[train_idx])
            y_pred = pipeline.predict(X_test)
            X = X_test
            y = y[test_idx]
            indices = np.asarray(test_idx, dtype=int)
        elif (cv_strategy == 'timeseries') or (name in ['rf', 'xgb', 'stacking', 'blending', 'rf+ann']) or isinstance(cv_obj, (GroupKFold, LeaveOneGroupOut)):
            print("使用手动KFold进行交叉验证（单线程）...")
            if isinstance(cv_obj, int):
                splitter = KFold(n_splits=cv_obj, shuffle=True, random_state=42)
            else:
                splitter = cv_obj

            if isinstance(splitter, (GroupKFold, LeaveOneGroupOut)):
                split_iter = splitter.split(X, y, groups)
            else:
                split_iter = splitter.split(X, y)

            y_pred = np.full_like(y, np.nan, dtype=float)
            total_folds = cv_splits if isinstance(cv_obj, int) else getattr(splitter, 'n_splits', cv_splits)

            for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
                print(f"Fold {fold_idx}/{total_folds}: 训练中...")
                X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
                X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
                fit_params = {}
                if groups is not None and name in ['blending', 'rf+ann']:
                    train_groups = np.array(groups)[train_idx]
                    n_train_groups = int(len(np.unique(train_groups)))
                    cv_use = max(2, min(cv_splits, n_train_groups)) if n_train_groups >= 2 else 2
                    pipeline.set_params(model__cv=cv_use)
                    fit_params = {'model__groups': train_groups}
                if groups is None and name in ['rf+ann', 'blending']:
                    pipeline.set_params(model__cv=max(3, cv_splits // 2))
                if name == 'xgb':
                    prep_steps = [
                        ('preproc', StableFeaturePreprocessor(
                            missing_ratio_threshold=0.95,
                            clip_quantiles=(0.01, 0.99),
                            variance_threshold=1e-12,
                            corr_threshold=0.85,
                            corr_method='pearson',
                            use_vif=False,
                            vif_threshold=10.0,
                            apply_power_transform=False,
                            scale_mode='robust'
                        ))
                    ]
                    if sel is not None:
                        prep_steps.append(('select', sel))
                    prep = Pipeline(prep_steps)
                    X_train_prep = prep.fit_transform(X_train, y[train_idx])
                    X_test_prep = prep.transform(X_test)
                    from sklearn.base import clone as _clone
                    est = _clone(model_clone)
                    try:
                        est.fit(X_train_prep, y[train_idx], eval_set=[(X_test_prep, y[test_idx])], early_stopping_rounds=80, verbose=0)
                    except TypeError:
                        est.fit(X_train_prep, y[train_idx])
                    y_pred[test_idx] = est.predict(X_test_prep)
                    best_iter = getattr(est, 'best_iteration', None)
                    if best_iter is not None:
                        print(f"Fold {fold_idx}: XGB 最佳迭代轮数={best_iter}")
                else:
                    pipeline.fit(X_train, y[train_idx], **fit_params)
                    y_pred[test_idx] = pipeline.predict(X_test)
                print(f"Fold {fold_idx}/{total_folds}: 完成")
            print("交叉验证完成！")
        else:
            y_pred = cross_val_predict(
                pipeline, X, y, cv=cv_obj, n_jobs=1, method='predict',
                groups=(groups if isinstance(cv_obj, (GroupKFold, LeaveOneGroupOut)) else None)
            )
            print("交叉验证完成！")
    except Exception as e:
        print(f"交叉验证失败: {e}")
        print("使用简单的训练测试分割作为备选...")
        idx_all = np.arange(len(y))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx_all, test_size=0.2, random_state=42
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y = y_test
        indices = idx_test

    # RUL任务预测值范围约束，避免线性模型产生极端值
    try:
        if str(target_type).lower() == 'rul':
            ymax = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else None
            if ymax is not None and np.isfinite(ymax):
                y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, ymax)
    except Exception:
        pass

    mask_valid = np.isfinite(y_pred)
    if not np.any(mask_valid):
        mask_valid = np.arange(len(y))
    y = np.asarray(y, dtype=float)[mask_valid]
    y_pred = np.asarray(y_pred, dtype=float)[mask_valid]
    try:
        if indices is not None and len(indices) == len(y_pred):
            indices = np.asarray(indices)[mask_valid]
        else:
            indices = np.where(mask_valid)[0]
    except Exception:
        indices = None
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))

    # 诊断：打印真实值与预测值的范围与尺度比
    try:
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        yp_min, yp_max = float(np.nanmin(y_pred)), float(np.nanmax(y_pred))
        scale_ratio = (np.nanmax(np.abs(y_pred)) + 1e-9) / (np.nanmax(np.abs(y)) + 1e-9)
        print(f"诊断: y[min,max,mean]=[{y_min:.3f},{y_max:.3f},{np.nanmean(y):.3f}]")
        print(f"诊断: y_pred[min,max,mean]=[{yp_min:.3f},{yp_max:.3f},{np.nanmean(y_pred):.3f}]")
        print(f"诊断: 预测尺度比 max|y_pred|/max|y| = {scale_ratio:.1f}x")
    except Exception:
        pass

    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        if (str(target_type).lower() == 'capacity_retention') and (data_mode == 'file') and os.path.isfile(xlsx_path):
            def _read_df(p):
                if p.lower().endswith('.csv'):
                    try:
                        return pd.read_csv(p, encoding='utf-8')
                    except Exception:
                        try:
                            return pd.read_csv(p, encoding='gbk')
                        except Exception:
                            return pd.read_csv(p, encoding='latin-1')
                from excel_loader import read_battery_excel
                s1, _, _ = read_battery_excel(p)
                return s1
            df_raw = _read_df(xlsx_path)
            cap_col, cycle_col, ret_col = detect_columns_en(df_raw)
            if cap_col is None:
                raise RuntimeError('Skip capacity_curve_ah: capacity column missing')
            if cycle_col is None:
                raise RuntimeError('Skip capacity_curve_ah: cycle column missing')
            df_use = df_raw.copy()
            df_use[cap_col] = pd.to_numeric(df_use[cap_col], errors='coerce')
            ser = df_use[[cycle_col, cap_col]].dropna().groupby(cycle_col, sort=True)[cap_col].mean()
            cycles_ah = ser.index.values
            true_ah = ser.values
            base_ah = float(true_ah[0]) if len(true_ah) > 0 else None
            if base_ah is None or not np.isfinite(base_ah):
                raise RuntimeError('Skip capacity_curve_ah: base capacity invalid')
            pct_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 100.0)
            pct_pred = pct_pred[:len(cycles_ah)]
            model_ah = np.clip(base_ah * pct_pred / 100.0, 0.0, base_ah * 1.2)
            eol_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
            eol_ah_env = os.environ.get('EOL_CAPACITY_AH', '').strip()
            eol_ah = float(eol_ah_env) if eol_ah_env else base_ah * eol_pct / 100.0
            fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            ax.plot(cycles_ah, true_ah, color='#4C72B0', linewidth=1.6, label=_label_text('True', '真实值', label_lang))
            ax.plot(cycles_ah, model_ah, color='#55A868', linewidth=1.6, linestyle='--', label=_label_text('Model', '模型', label_lang))
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('Capacity (Ah)', '容量(Ah)', label_lang))
            ymin = max(0.0, eol_ah * 0.9)
            ymax = max(base_ah * 1.05, eol_ah * 1.05)
            ax.set_xlim(float(np.min(cycles_ah)), float(np.max(cycles_ah)))
            ax.set_ylim(ymin, ymax)
            ax.margins(x=0.02, y=0.05)
            try:
                ax.axhline(eol_ah, color='gray', linestyle=':', linewidth=1)
            except Exception:
                pass
            ax.legend(); ax.grid(True, alpha=0.3)
            _save_plot(fig, os.path.join(art_dir, 'capacity_curve_ah'))
            try:
                pd.DataFrame({'cycle': cycles_ah, 'capacity_true_ah': true_ah, 'capacity_model_ah': model_ah, 'eol_capacity_ah': [eol_ah]*len(cycles_ah)}).to_csv(
                    os.path.join(art_dir, 'capacity_curve_ah.csv'), index=False, encoding='utf-8-sig')
            except Exception:
                pass
    except Exception:
        pass

    print(f"模型 {name} 评估完成: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    out = {'name': name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred': y_pred, 'y_true': y, 'pipeline': pipeline}
    try:
        out['indices'] = indices if 'indices' in locals() else None
    except Exception:
        out['indices'] = None
    return out

def compute_feature_corr_df(X_df, target_vec, method='spearman'):
    if not isinstance(X_df, pd.DataFrame):
        _arr = np.asarray(X_df)
        if _arr.ndim == 1:
            _arr = _arr.reshape(-1, 1)
        X_df = pd.DataFrame(_arr, columns=[f'f_{i}' for i in range(_arr.shape[1])])

    corr_list = []
    for col in X_df.columns:
        try:
            if method == 'spearman':
                from scipy.stats import spearmanr
                corr_val, p_val = spearmanr(X_df[col], target_vec, nan_policy='omit')
            elif method == 'pearson':
                from scipy.stats import pearsonr
                corr_val, p_val = pearsonr(X_df[col], target_vec)
            else:
                corr_val, p_val = 0.0, 1.0
            corr_list.append({'feature': col, 'correlation': float(corr_val) if not np.isnan(corr_val) else 0.0, 'p_value': float(p_val) if not np.isnan(p_val) else 1.0})
        except Exception:
            corr_list.append({'feature': col, 'correlation': 0.0, 'p_value': 1.0})

    return pd.DataFrame(corr_list).sort_values('correlation', key=abs, ascending=False)

def save_corr_heatmap(corr_df, title, save_path, top_k=50):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        top_corr = corr_df.head(top_k)
        corr_matrix = top_corr.set_index('feature')['correlation'].to_frame().T
        plt.figure(figsize=(16, 8), constrained_layout=True)
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f', cbar_kws={'label': 'Correlation'})
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
        except Exception:
            pass
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save heatmap {save_path}: {e}")

def save_literature_style_corr_matrix(X_df, target_vec, title, save_path, method='spearman', top_k=13, include_true=None, label_lang='auto'):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        corr_df = compute_feature_corr_df(X_df, target_vec, method=method)
        top_features = corr_df.head(top_k)['feature'].tolist()
        if include_true is not None and len(include_true) > 0:
            top_features = list(set(top_features + include_true))
        if isinstance(X_df, pd.DataFrame):
            _use_cols = [c for c in top_features if c in X_df.columns]
            X_subset = X_df[_use_cols]
        else:
            _arr = np.asarray(X_df)
            if _arr.ndim == 1:
                _arr = _arr.reshape(-1, 1)
            _df_tmp = pd.DataFrame(_arr, columns=[f'f_{i}' for i in range(_arr.shape[1])])
            _use_cols = [c for c in top_features if c in _df_tmp.columns]
            X_subset = _df_tmp[_use_cols]
        corr_matrix = X_subset.corr(method=method)
        plt.figure(figsize=(12, 10), constrained_layout=True)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            plt.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
        except Exception:
            pass
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save correlation matrix {save_path}: {e}")

def save_shap_analysis(selector, X, y, art_dir):
    """保存SHAP分析结果与条形图"""
    try:
        import shap
    except ImportError:
        print("\n⚠️  SHAP模块未安装，跳过SHAP分析")
        print("提示：可使用 'pip install shap' 安装SHAP模块以启用特征重要性解释")
        return
    
    try:
        import matplotlib.pyplot as plt
        from xgboost import XGBRegressor
        if not hasattr(selector, 'selected_features_') or not getattr(selector, 'selected_features_', []):
            return
        X_sel = selector.transform(X)
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_sel, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sel)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sel, feature_names=selector.selected_features_, show=False)
        plt.savefig(os.path.join(art_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        try:
            plt.savefig(os.path.join(art_dir, 'shap_summary.svg'), bbox_inches='tight')
        except Exception:
            pass
        plt.close()
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sel, feature_names=selector.selected_features_, plot_type='bar', show=False)
        plt.savefig(os.path.join(art_dir, 'shap_importance_bar.png'), dpi=300, bbox_inches='tight')
        try:
            plt.savefig(os.path.join(art_dir, 'shap_importance_bar.svg'), bbox_inches='tight')
        except Exception:
            pass
        plt.close()
        print("✅ SHAP分析完成，结果已保存")
    except Exception as e:
        print(f"Warning: SHAP 分析失败，已跳过: {e}")

# 通用列检测（容量/循环/保持率）
def detect_columns(df):
    import re
    cols = list(df.columns)
    cap_ah_col, cycle_col, ret_pct_col = None, None, None
    for c in cols:
        name = str(c)
        if cap_ah_col is None:
            if re.search(r"(?i)\(\s*Ah\s*\)", name) or (('容量' in name) and re.search(r"(?i)Ah", name)):
                cap_ah_col = c
        if cycle_col is None:
            if ('循环' in name) or re.search(r"(?i)cycle", name):
                cycle_col = c
        if ret_pct_col is None:
            if ('保持' in name) or ('保持率' in name) or re.search(r"(?i)retention", name) or ('%' in name) or re.search(r"(?i)pct", name):
                ret_pct_col = c
    return cap_ah_col, cycle_col, ret_pct_col

def detect_columns_en(df):
    import re
    cols = list(df.columns)
    cap_ah_col, cycle_col, ret_pct_col = None, None, None
    for c in cols:
        name = str(c)
        if cap_ah_col is None:
            if (name.lower() in ('discharge_capacity_ah','capacity_ah')) or re.search(r"(?i)discharge\s*capacity", name):
                cap_ah_col = c
        if cycle_col is None:
            if (name.lower() == 'cycle'):
                cycle_col = c
        if ret_pct_col is None:
            if (name.lower() == 'capacity_retention_pct') or re.search(r"(?i)retention", name) or re.search(r"(?i)pct", name):
                ret_pct_col = c
    return cap_ah_col, cycle_col, ret_pct_col

def plot_capacity_compare_ah(xlsx_path, data_mode, y_pred_oof, art_dir, label_lang, keys_df=None, y_true_used=None, pred_label='Model'):
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        # 若直接提供了真实保持率序列，则优先生成“保持率(%)”对比图（无需读取原始文件）
        if y_true_used is not None and len(np.asarray(y_true_used)) > 0:
            cycles = None
            if isinstance(keys_df, pd.DataFrame) and 'cycle' in keys_df.columns:
                cycles = keys_df['cycle'].values
            else:
                cycles = np.arange(len(y_true_used))
            true_pct = np.clip(np.asarray(y_true_used, dtype=float)[:len(cycles)], 0.0, 120.0)
            pred_pct = np.clip(np.asarray(y_pred_oof, dtype=float)[:len(cycles)], 0.0, 120.0)
            eol_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
            fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            ax.plot(cycles, pred_pct, color='black', linewidth=1.6, marker='s', markersize=4, label=_label_text(pred_label, pred_label, label_lang))
            ax.plot(cycles, true_pct, color='black', linewidth=1.6, linestyle=':', label=_label_text('Original', '原始值', label_lang))
            try:
                ax.axhline(eol_pct, color='black', linestyle='--', linewidth=1.2, label=_label_text('EOL', '失效阈值', label_lang))
            except Exception:
                pass
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('Capacity Retention (%)', '容量保持率(%)', label_lang))
            ax.set_xlim(float(np.min(cycles)), float(np.max(cycles)))
            y_top = float(max(100.0, 1.05 * max(np.nanmax(true_pct), np.nanmax(pred_pct))))
            ax.set_ylim(0.0, y_top)
            ax.grid(True, alpha=0.3)
            ax.legend(); ax.margins(x=0.02, y=0.05)
            _save_plot(fig, os.path.join(art_dir, 'capacity_compare_pct'))
            try:
                pd.DataFrame({'cycle': cycles, 'capacity_true_pct': true_pct, 'capacity_pred_pct': pred_pct, 'eol_capacity_pct': [eol_pct]*len(cycles)}).to_csv(
                    os.path.join(art_dir, 'capacity_compare_pct.csv'), index=False, encoding='utf-8-sig')
            except Exception:
                pass
            return True
        if data_mode != 'file' or not os.path.isfile(xlsx_path):
            return False
        def _read_df(p):
            if p.lower().endswith('.csv'):
                try:
                    return pd.read_csv(p, encoding='utf-8')
                except Exception:
                    try:
                        return pd.read_csv(p, encoding='gbk')
                    except Exception:
                        return pd.read_csv(p, encoding='latin-1')
            from excel_loader import read_battery_excel
            s1, _, _ = read_battery_excel(p)
            return s1
        df = _read_df(xlsx_path)
        cap_col, cycle_col, ret_col = detect_columns_en(df)
        if ret_col is not None and (cap_col is None):
            df_use = df.copy()
            df_use[ret_col] = pd.to_numeric(df_use[ret_col], errors='coerce')
            if cycle_col is not None:
                ser = df_use[[cycle_col, ret_col]].dropna().groupby(cycle_col, sort=True)[ret_col].mean()
                cycles = ser.index.values
                true_pct = ser.values
            elif isinstance(keys_df, pd.DataFrame) and 'cycle' in keys_df.columns:
                cycles = keys_df['cycle'].values
                true_pct = df_use[ret_col].dropna().values[:len(cycles)]
            else:
                return False
            pred_pct = np.clip(np.asarray(y_pred_oof, dtype=float)[:len(cycles)], 0.0, 120.0)
            eol_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
            fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            ax.plot(cycles, pred_pct, color='black', linewidth=1.6, marker='s', markersize=4, label=_label_text(pred_label, pred_label, label_lang))
            ax.plot(cycles, true_pct, color='black', linewidth=1.6, linestyle=':', label=_label_text('Original', '原始值', label_lang))
            try:
                ax.axhline(eol_pct, color='black', linestyle='--', linewidth=1.2, label=_label_text('EOL', '失效阈值', label_lang))
            except Exception:
                pass
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('Capacity Retention (%)', '容量保持率(%)', label_lang))
            ax.set_xlim(float(np.min(cycles)), float(np.max(cycles)))
            y_top = float(max(100.0, 1.05 * max(np.nanmax(true_pct), np.nanmax(pred_pct))))
            ax.set_ylim(0.0, y_top)
            ax.grid(True, alpha=0.3)
            ax.legend(); ax.margins(x=0.02, y=0.05)
            _save_plot(fig, os.path.join(art_dir, 'capacity_compare_pct'))
            try:
                pd.DataFrame({'cycle': cycles, 'capacity_true_pct': true_pct, 'capacity_pred_pct': pred_pct, 'eol_capacity_pct': [eol_pct]*len(cycles)}).to_csv(
                    os.path.join(art_dir, 'capacity_compare_pct.csv'), index=False, encoding='utf-8-sig')
            except Exception:
                pass
            return True
        elif cap_col is not None:
            df_use = df.copy()
            df_use[cap_col] = pd.to_numeric(df_use[cap_col], errors='coerce')
            if cycle_col is not None:
                ser = df_use[[cycle_col, cap_col]].dropna().groupby(cycle_col, sort=True)[cap_col].mean()
                cycles = ser.index.values
                true_ah = ser.values
                m_valid = np.isfinite(true_ah) & (true_ah > 0.1)
                cycles = cycles[m_valid]
                true_ah = true_ah[m_valid]
            elif isinstance(keys_df, pd.DataFrame) and 'cycle' in keys_df.columns:
                cycles = keys_df['cycle'].values
                true_ah = df_use[cap_col].dropna().values[:len(cycles)]
            else:
                return False
            if len(true_ah) == 0 or not np.isfinite(true_ah[0]):
                return False
            base_ah = float(true_ah[0])
            pct_pred = np.clip(np.asarray(y_pred_oof, dtype=float)[:len(cycles)], 0.0, 120.0)
            pred_ah = np.clip(base_ah * pct_pred / 100.0, 0.0, base_ah * 1.2)
            eol_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
            eol_ah = base_ah * eol_pct / 100.0
            fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            ax.plot(cycles, pred_ah, color='black', linewidth=1.6, marker='s', markersize=4, label=_label_text(pred_label, pred_label, label_lang))
            ax.plot(cycles, true_ah, color='black', linewidth=1.6, linestyle=':', label=_label_text('Original', '原始值', label_lang))
            try:
                ax.axhline(eol_ah, color='black', linestyle='--', linewidth=1.2, label=_label_text('EOL', '失效阈值', label_lang))
            except Exception:
                pass
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('Capacity (Ah)', '容量(Ah)', label_lang))
            ax.set_xlim(float(np.min(cycles)), float(np.max(cycles)))
            y_max = float(max(np.nanmax(true_ah), np.nanmax(pred_ah)))
            ax.set_ylim(max(0.0, eol_ah * 0.9), y_max * 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(); ax.margins(x=0.02, y=0.05)
            _save_plot(fig, os.path.join(art_dir, 'capacity_compare_ah'))
            try:
                pd.DataFrame({'cycle': cycles, 'capacity_true_ah': true_ah, 'capacity_pred_ah': pred_ah, 'eol_capacity_ah': [eol_ah]*len(cycles)}).to_csv(
                    os.path.join(art_dir, 'capacity_compare_ah.csv'), index=False, encoding='utf-8-sig')
            except Exception:
                pass
            return True
        else:
            return False
    except Exception:
        return False

def build_dataset_from_directory(dir_path: Optional[str] = None, target_type: str = 'RUL', capacity_threshold_pct: float = 80.0, return_keys: bool = False):
    import logging
    logger = logging.getLogger(__name__)

    if dir_path is None or str(dir_path).strip() == '':
        env_dir = os.environ.get('BATTERY_XLSX_DIR', '').strip()
        if env_dir:
            dir_path = os.path.abspath(env_dir)
        else:
            try:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            except Exception:
                project_root = os.getcwd()
            dir_path = os.path.join(project_root, 'data')
    else:
        dir_path = os.path.abspath(dir_path)

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f'目录不存在：{dir_path}')

    print(f'使用数据目录：{dir_path}')

    xlsx_list = sorted(glob.glob(os.path.join(dir_path, '**', '*.xlsx'), recursive=True))
    csv_list = sorted(glob.glob(os.path.join(dir_path, '**', '*.csv'), recursive=True))
    xlsx_list = [p for p in xlsx_list if not os.path.basename(p).startswith('~$')]
    csv_list = [p for p in csv_list if not os.path.basename(p).startswith('~$')]
    file_list = xlsx_list + csv_list

    print(f'递归检索完成：找到 {len(file_list)} 个文件')
    if file_list[:5]:
        rel_samples = [os.path.relpath(p, dir_path) for p in file_list[:5]]
        print(f'示例文件：{rel_samples}')

    if not file_list:
        raise ValueError(
            f'目录中未找到 .xlsx 或 .csv 文件（递归检索）：{dir_path}\n'
            f'请检查环境变量 BATTERY_XLSX_DIR 或确认路径包含数据文件'
        )

    X_list, y_list, groups_list, keys_list = [], [], [], []
    cols_union = set()

    for p in file_list:
        try:
            from build_dataset import build_dataset_from_excel as _bdf
            if return_keys:
                Xi, yi, keys_df = _bdf(p, target_type=target_type, capacity_threshold_pct=capacity_threshold_pct, return_keys=True)
            else:
                Xi, yi = _bdf(p, target_type=target_type, capacity_threshold_pct=capacity_threshold_pct)
            if not isinstance(Xi, pd.DataFrame):
                Xi = pd.DataFrame(Xi)
            cols_union |= set(Xi.columns)
            X_list.append(Xi)
            y_list.append(np.asarray(yi))
            groups_list.extend([os.path.basename(p)] * len(yi))
            if return_keys:
                kd = keys_df.copy() if isinstance(keys_df, pd.DataFrame) else pd.DataFrame(index=Xi.index)
                keys_list.append(kd)
            logger.info(f'Read {os.path.basename(p)}: X={Xi.shape}, y={len(yi)}')
        except Exception as e:
            logger.warning(f'Skip {p}: {e}')

    if not X_list:
        raise ValueError('目录中没有有效的 Excel 或 CSV 数据用于构建数据集')

    all_cols = sorted(cols_union)
    X_aligned = [Xi.reindex(columns=all_cols) for Xi in X_list]

    print('开始数据合并...')
    X_all = pd.concat(X_aligned, ignore_index=True)
    y_all = np.concatenate(y_list)
    groups = np.array(groups_list, dtype=object)
    keys_all = None
    if return_keys and keys_list:
        try:
            keys_all = pd.concat(keys_list, ignore_index=True)
        except Exception:
            keys_all = None
    print(f'合并完成：X_all.shape={X_all.shape}, y_all.shape={y_all.shape}, 分组数量={len(np.unique(groups))}')

    if return_keys:
        return X_all, y_all, groups, keys_all
    return X_all, y_all, groups

def preclean_features(X, y, corr_threshold=0.85, vif_threshold=10.0):
    import numpy as np
    import pandas as pd
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    df_dedup = df.T.drop_duplicates().T
    removed_dup = [c for c in df.columns if c not in df_dedup.columns]
    # 统一数值化 + 替换非有限值，用于相关性与后续VIF
    df_num = df_dedup.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    med = df_num.median()
    df_num = df_num.fillna(med)
    def _safe_corr(a, b):
        try:
            c = np.corrcoef(a, b)[0, 1]
            return 0.0 if np.isnan(c) else float(c)
        except Exception:
            return 0.0
    target_corr = df_num.apply(lambda col: _safe_corr(col.values, np.asarray(y)), axis=0)
    order = target_corr.abs().sort_values(ascending=False).index.tolist()
    corr_abs = df_num.corr(method='pearson').abs()
    keep = []
    for f in order:
        if f not in corr_abs.columns:
            continue
        if any(corr_abs.loc[f, k] >= corr_threshold for k in keep):
            continue
        keep.append(f)
    dropped_corr = [c for c in df_dedup.columns if c not in keep]
    # 关键：VIF 使用已经数值化/填充的 df_num 子集
    df_reduced = df_num[keep]
    dropped_vif = []
    # 如果阈值为非有限（如 inf），明确跳过VIF计算
    if not np.isfinite(float(vif_threshold)):
        df_vif = df_reduced
        return df_vif, {
            'removed_exact_duplicates': removed_dup,
            'removed_high_corr': dropped_corr,
            'removed_high_vif': dropped_vif
        }
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        arr = df_reduced.values.astype('float64')
        # 保护：若仍有非有限值，跳过VIF
        if not np.isfinite(arr).all():
            df_vif = df_reduced
        else:
            cols = list(df_reduced.columns)
            while arr.shape[1] >= 2:
                vifs = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
                max_vif = float(np.nanmax(vifs))
                if not np.isfinite(max_vif) or max_vif <= float(vif_threshold):
                    break
                idx = int(np.nanargmax(vifs))
                dropped_vif.append(cols[idx])
                cols.pop(idx)
                arr = np.delete(arr, idx, axis=1)
            df_vif = df_reduced[cols]
    except Exception as e:
        print(f"VIF 步骤跳过（statsmodels不可用或计算失败）: {e}")
        df_vif = df_reduced
    return df_vif, {
        'removed_exact_duplicates': removed_dup,
        'removed_high_corr': dropped_corr,
        'removed_high_vif': dropped_vif
    }

# 规范实现：保留唯一实现（返回去重后的 DataFrame 与被移除列名列表）
def _drop_duplicate_columns_by_values_canonical(X_df):
    import numpy as np
    import pandas as pd
    cols = list(X_df.columns)
    seen = set()
    keep = []
    for c in cols:
        s = pd.to_numeric(X_df[c], errors='coerce')
        arr = s.values.astype('float64')
        arr = np.where(np.isnan(arr), -999999.0, arr)
        h = hashlib.sha1(arr.tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(c)
    removed = [c for c in cols if c not in keep]
    return X_df[keep], removed


def _drop_highly_correlated_features(X_df, y, corr_threshold=0.95, method='pearson'):
    """
    移除高度线性相关的特征，避免多重共线性。
    
    策略：
    1. 计算所有特征对之间的相关系数
    2. 对于|corr| > threshold的特征对，保留与目标变量相关性更强的那个
    3. 返回移除后的DataFrame和被移除的列名列表
    
    Args:
        X_df: 特征DataFrame
        y: 目标变量
        corr_threshold: 相关系数阈值，默认0.95
        method: 相关系数计算方法，'pearson' 或 'spearman'
    
    Returns:
        (X_filtered, removed_cols): 过滤后的DataFrame和被移除的列名列表
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    
    print(f"\n开始多重共线性检测（相关系数阈值: {corr_threshold}）...")
    
    # 转换为数值类型
    X_numeric = X_df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    
    # 计算特征之间的相关矩阵
    if method == 'spearman':
        corr_matrix = X_numeric.corr(method='spearman')
    else:
        corr_matrix = X_numeric.corr(method='pearson')
    
    # 计算每个特征与目标变量的相关性
    y_numeric = pd.to_numeric(pd.Series(y), errors='coerce')
    target_corr = {}
    for col in X_numeric.columns:
        try:
            if method == 'spearman':
                corr_val, _ = spearmanr(X_numeric[col].fillna(X_numeric[col].median()), 
                                       y_numeric.fillna(y_numeric.median()))
            else:
                corr_val, _ = pearsonr(X_numeric[col].fillna(X_numeric[col].median()), 
                                      y_numeric.fillna(y_numeric.median()))
            target_corr[col] = abs(corr_val) if not np.isnan(corr_val) else 0.0
        except Exception:
            target_corr[col] = 0.0
    
    # 找出高度相关的特征对
    cols_to_drop = set()
    cols = list(corr_matrix.columns)
    high_corr_pairs = []
    
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            col_i, col_j = cols[i], cols[j]
            if col_i in cols_to_drop or col_j in cols_to_drop:
                continue
            
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > corr_threshold:
                # 对于高度相关的特征对，保留与目标相关性更强的那个
                corr_i_target = target_corr.get(col_i, 0.0)
                corr_j_target = target_corr.get(col_j, 0.0)
                
                if corr_i_target >= corr_j_target:
                    to_drop = col_j
                    to_keep = col_i
                else:
                    to_drop = col_i
                    to_keep = col_j
                
                cols_to_drop.add(to_drop)
                high_corr_pairs.append({
                    'feature_1': col_i,
                    'feature_2': col_j,
                    'correlation': corr_val,
                    'dropped': to_drop,
                    'kept': to_keep,
                    f'{to_keep}_target_corr': target_corr.get(to_keep, 0.0),
                    f'{to_drop}_target_corr': target_corr.get(to_drop, 0.0)
                })
    
    removed_cols = list(cols_to_drop)
    X_filtered = X_df.drop(columns=removed_cols)
    
    if len(removed_cols) > 0:
        print(f"✅ 移除了 {len(removed_cols)} 个高度相关的特征（{X_df.shape[1]} -> {X_filtered.shape[1]}）")
        print(f"被移除的特征: {removed_cols[:10]}{'...' if len(removed_cols) > 10 else ''}")
        if len(high_corr_pairs) > 0:
            print(f"\n高度相关特征对（前5个）:")
            for pair in high_corr_pairs[:5]:
                kept_corr_key = pair['kept'] + '_target_corr'
                dropped_corr_key = pair['dropped'] + '_target_corr'
                print(f"  {pair['feature_1']} <-> {pair['feature_2']}: "
                      f"corr={pair['correlation']:.3f}, "
                      f"保留={pair['kept']}(与目标corr={pair[kept_corr_key]:.3f}), "
                      f"移除={pair['dropped']}(与目标corr={pair[dropped_corr_key]:.3f})")
    else:
        print(f"✅ 未发现高度相关的特征对（阈值 > {corr_threshold}）")
    
    return X_filtered, removed_cols

# 兼容旧接口的轻量包装
def _drop_duplicate_columns_by_values(X_df):
    return _drop_duplicate_columns_by_values_canonical(X_df)

# 顶层新增：Stacking 诊断函数
def diagnose_stacking(stacking, X_selected, y):
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    import numpy as np
    import pandas as pd
    try:
        stack_fit = clone(stacking)
        stack_fit.fit(X_selected, y)
        names = [name for name, _ in stack_fit.estimators]
        print("=== 基学习器预测范围统计 ===")
        base_preds = []
        for name, est in zip(names, stack_fit.estimators_):
            pred = est.predict(X_selected)
            base_preds.append(pred)
            pmin, pmax, pmean = float(np.nanmin(pred)), float(np.nanmax(pred)), float(np.nanmean(pred))
            pstd = float(np.nanstd(pred))
            print(f"{name}: min={pmin:.6f}, max={pmax:.6f}, mean={pmean:.6f}, std={pstd:.6f}")
        stack_pred = stack_fit.predict(X_selected)
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        sp_min, sp_max, sp_mean = float(np.nanmin(stack_pred)), float(np.nanmax(stack_pred)), float(np.nanmean(stack_pred))
        scale_ratio = (np.nanmax(np.abs(stack_pred)) + 1e-9) / (np.nanmax(np.abs(y)) + 1e-9)
        print("=== Stacking 整体预测范围（非 OOF，仅诊断）===")
        print(f"stack_pred: min={sp_min:.6f}, max={sp_max:.6f}, mean={sp_mean:.6f}, scale_ratio={scale_ratio:.1f}x")
        print(f"y: min={y_min:.6f}, max={y_max:.6f}, mean={float(np.nanmean(y)):.6f}")
        pred_X = np.column_stack(base_preds)
        meta_est = clone(stack_fit.final_estimator_)
        meta_est.fit(pred_X, y)
        meta_model = getattr(meta_est, 'named_steps', {}).get('enet', meta_est)
        if hasattr(meta_model, 'alpha_'):
            print(f"元学习器 alpha_ = {float(meta_model.alpha_):.6f}")
        coefs = np.asarray(getattr(meta_model, 'coef_', np.full(len(names), np.nan))).ravel()
        print("=== 元学习器系数（对应各基学习器）===")
        for name, c in zip(names, coefs):
            print(f"{name}: coef={float(c):.6f}")
        perm = permutation_importance(meta_est, pred_X, y, scoring='neg_mean_squared_error', n_repeats=5, random_state=42)
        print("=== 置换重要性（负均方误差越大越重要）===")
        for name, p in zip(names, perm.importances_mean):
            print(f"{name}: perm_mean={float(p):.6f}")
        _arr = np.asarray(pred_X)
        if _arr.ndim == 1:
            _arr = _arr.reshape(-1, 1)
        _cols = list(names)
        if _arr.shape[1] != len(_cols):
            _cols = [f'm_{i}' for i in range(_arr.shape[1])]
        pred_df = pd.DataFrame(_arr, columns=_cols)
        print("=== 基学习器预测两两相关性 ===")
        print(pred_df.corr().round(4))
    except Exception as e:
        print(f"diagnose_stacking 诊断失败: {e}")

if __name__ == '__main__':
    print('=== 训练入口已启动 ===')
    RUN_ONLY_STACKING = False

    # 环境与日志
    xlsx_path = os.environ.get('BATTERY_XLSX_PATH', 'data/battery.xlsx')
    target_type = os.environ.get('TARGET_TYPE', 'RUL')
    threshold_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
    cv_splits = int(os.environ.get('CV_SPLITS', '8'))
    cv_strategy = os.environ.get('CV_STRATEGY', 'kfold').lower()
    import logging
    import sys

    art_dir = os.environ.get('ARTIFACTS_DIR', os.path.join(os.getcwd(), 'artifacts'))
    os.makedirs(art_dir, exist_ok=True)
    log_path = os.path.join(art_dir, 'training.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    try:
        import random
        random.seed(42)
    except Exception:
        pass
    try:
        np.random.seed(42)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
    except Exception:
        pass

    # 一键复现模式：设置合理默认
    repro = os.environ.get('REPRO_MODE', '').strip().lower()
    if repro in ('1','true','on'):
        os.environ.setdefault('CV_STRATEGY', 'group_holdout')
        os.environ.setdefault('GROUP_HOLDOUT_RATIO', '0.8')
        os.environ.setdefault('GROUP_HOLDOUT_SEED', '42')
        os.environ.setdefault('OVERLAY_MODELS', 'ridge,svr,xgb,stacking,blending')
        os.environ.setdefault('LABEL_LANG', 'en')
        print('REPRO_MODE: 已应用默认参数（group_holdout 80/20, seed=42）')

    # 数据构建
    xlsx_dir = os.environ.get('BATTERY_XLSX_DIR', '').strip()
    print("开始构建数据集...")
    groups = None
    if xlsx_dir and os.path.isdir(xlsx_dir):
        X, y, groups, keys_df = build_dataset_from_directory(xlsx_dir, target_type=target_type, capacity_threshold_pct=threshold_pct, return_keys=True)
        data_mode = 'dir'
    else:
        print(f"使用文件模式，文件: {xlsx_path}")
        X, y, keys_df = build_dataset_from_excel(xlsx_path, target_type=target_type, capacity_threshold_pct=threshold_pct, return_keys=True)
        data_mode = 'file'
        print(f"文件模式数据加载完成: X.shape={X.shape}, y.shape={y.shape}")
    if groups is not None and (os.environ.get('CV_STRATEGY', '').strip() == '' or cv_strategy in ('', 'kfold')):
        # 更符合你要求的默认：按电池组留出
        cv_strategy = 'group_holdout'
        print('交叉验证策略自动设置为 group_holdout (80/20)')
    elif os.environ.get('CV_STRATEGY', '').strip() == '' or cv_strategy in ('', 'kfold'):
        if str(target_type).lower() in ('capacity_retention', 'rul'):
            cv_strategy = 'timeseries'
            print('交叉验证策略自动设置为 timeseries')
    # 强化RUL/保持率任务的策略约束：使用电池分组留出
    if str(target_type).lower() in ('capacity_retention', 'rul'):
        # RUL任务必须使用电池分组策略，禁止KFold
        allowed = {'group_holdout','groupkfold','group_kfold','logo','leaveonegroupout','leave_one_group_out'}
        if cv_strategy not in allowed:
            if groups is not None and len(pd.unique(groups)) > 3:
                print(f"RUL/保持率任务不允许使用 {cv_strategy}，强制改为 group_holdout（按电池80/20分组留出）")
                cv_strategy = 'group_holdout'
            else:
                print(f"RUL/保持率任务不允许使用 {cv_strategy}，强制改为 timeseries")
                cv_strategy = 'timeseries'

    # RUL 任务：使用电池分组留出策略（而非时序切分）
    # 前80%电池用于训练，后20%电池用于测试（保留完整生命周期）
    PREDEFINED_SPLIT = None  # 初始化全局变量，避免NameError
    try:
        if str(target_type).lower() == 'rul' and groups is not None and len(pd.unique(groups)) > 3:
            # 优先使用group_holdout策略（已在CV策略中处理）
            # 这里不再需要PREDEFINED_SPLIT，让group_holdout自然分组
            print(f"RUL任务将使用 group_holdout 策略：前{int(float(os.environ.get('GROUP_HOLDOUT_RATIO', '0.8'))*100)}%电池训练，后{int((1-float(os.environ.get('GROUP_HOLDOUT_RATIO', '0.8')))*100)}%电池测试")
        elif str(target_type).lower() == 'rul' and isinstance(keys_df, pd.DataFrame) and 'cycle' in keys_df.columns:
            # 回退：按每个电池时序切分（前N%训练，后(100-N)%测试）
            ratio = float(os.environ.get('RUL_TRAIN_RATIO', '0.6'))
            if ratio <= 0.0 or ratio >= 1.0:
                ratio = 0.6
            train_idx, test_idx = [], []
            if '条码' in keys_df.columns:
                id_vals = keys_df['条码'].astype(str).values
            else:
                try:
                    id_vals = np.asarray(groups).astype(str) if groups is not None else np.array(['group'] * len(keys_df), dtype=object)
                except Exception:
                    id_vals = np.array(['group'] * len(keys_df), dtype=object)
            cyc_vals = np.asarray(keys_df['cycle'].values, dtype=float)
            for uid in pd.unique(id_vals):
                m = (id_vals == uid)
                idx_all = np.where(m)[0]
                if idx_all.size == 0:
                    continue
                order = np.argsort(cyc_vals[idx_all], kind='mergesort')
                idx_sorted = idx_all[order]
                n = idx_sorted.size
                k = max(1, int(np.floor(n * ratio)))
                train_idx.extend(list(idx_sorted[:k]))
                test_idx.extend(list(idx_sorted[k:]))
            if len(test_idx) >= 2 and len(train_idx) >= 2:
                PREDEFINED_SPLIT = (np.array(train_idx, dtype=int), np.array(test_idx, dtype=int))
                print(f"已创建按电池时序划分（每电池前{int(ratio*100)}%训练）：train={len(train_idx)}, test={len(test_idx)}")
    except Exception as e:
        print(f"按电池时序划分创建失败: {e}")

    # 训练前：剔除"值完全相同"的重复列（使用规范实现）
    before = X.shape[1]
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    X, removed_dup_cols = _drop_duplicate_columns_by_values_canonical(X_df)
    after = X.shape[1]
    if after < before:
        print(f"移除了 {before - after} 个值完全相同的重复列（{before} -> {after}）")
        print(f"重复列名: {removed_dup_cols}")
    
    # 训练前：剔除高度线性相关的特征（避免多重共线性）
    import os
    corr_threshold = float(os.environ.get('MULTICOLLINEARITY_THRESHOLD', '0.95'))
    before_corr = X.shape[1]
    X, removed_corr_cols = _drop_highly_correlated_features(X, y, corr_threshold=corr_threshold, method='pearson')
    after_corr = X.shape[1]
    if len(removed_corr_cols) > 0:
        print(f"多重共线性处理完成：{before_corr} -> {after_corr} 个特征")

    print(f"开始特征选择，原始特征数: {X.shape[1]}")
    sample_count = X.shape[0]
    feature_count = X.shape[1]
    # 动态提升选特征上限，避免每折只剩极少特征
    if sample_count < 500:
        max_features = max(12, min(30, int(sample_count * 0.2)))
    elif sample_count < 1000:
        max_features = max(20, min(60, int(sample_count * 0.12)))
    else:
        max_features = max(25, min(80, int(sample_count * 0.10)))
    print(f"根据样本数 {sample_count} 调整特征数为: {max_features}")
    selector = CombinedFeatureSelector(
        n_features=max_features,
        relief_neighbors=min(50, sample_count//10),
        correlation_threshold=0.85,  # 该参数在选择器内部作为兜底，特征对相关性在此之前已处理
        random_state=42,
        weights=(0.33, 0.33, 0.34)
    )
    print("特征选择器创建完成，开始特征选择过程...")
    print("正在进行特征选择...")
    X_selected = selector.fit_transform(X, y)
if __name__ == '__main__':
    print(f"特征选择完成: {X.shape[1]} -> {X_selected.shape[1]} 个特征")
    if hasattr(selector, 'selected_features_') and len(getattr(selector, 'selected_features_', [])) > 0:
        save_shap_analysis(selector, X, y, art_dir)

    # 保存每路重要性明细（便于论文）
    try:
        feat_names = list(getattr(selector, 'feature_names_', []))
        df_imp = pd.DataFrame({
            'feature': feat_names,
            'relief': list(getattr(selector, 'imp_rel_', np.zeros(len(feat_names))))
            if hasattr(selector, 'imp_rel_') else [0.0] * len(feat_names),
            'shap_or_perm': list(getattr(selector, 'imp_shap_', np.zeros(len(feat_names))))
            if hasattr(selector, 'imp_shap_') else [0.0] * len(feat_names),
            'lasso': list(getattr(selector, 'imp_lasso_', np.zeros(len(feat_names))))
            if hasattr(selector, 'imp_lasso_') else [0.0] * len(feat_names),
            'corr': list(getattr(selector, 'corr_importance_', np.zeros(len(feat_names))))
            if hasattr(selector, 'corr_importance_') else [0.0] * len(feat_names),
            'agg': list(getattr(selector, 'aggregated_importance_', np.zeros(len(feat_names))))
            if hasattr(selector, 'aggregated_importance_') else [0.0] * len(feat_names)
        })
        df_imp.sort_values('agg', ascending=False).to_csv(
            os.path.join(art_dir, 'feature_importance_breakdown.csv'),
            index=False, encoding='utf-8-sig'
        )
    except Exception:
        pass

    print("开始选择基础模型...")
    base = choose_base_models_by_data(X_selected, y, random_state=42, cv_splits=cv_splits)
    print(f"选择了 {len(base)} 个基础模型: {list(base.keys())}")
    if 'torch_mlp' in base:
        print("✓ torch_mlp 已包含在基础模型中")
        print(f"torch_mlp 设备: {base['torch_mlp'].device}")
    else:
        print("✗ torch_mlp 未包含在基础模型中")
        print(f"TORCH_AVAILABLE: {TORCH_AVAILABLE}")
        if TORCH_AVAILABLE:
            import torch as _torch
            print(f"PyTorch CUDA 可用: {_torch.cuda.is_available()}")
            if _torch.cuda.is_available():
                print(f"GPU 数量: {_torch.cuda.device_count()}")
                for i in range(_torch.cuda.device_count()):
                    print(f"GPU {i}: {_torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch 未安装或导入失败")

    print("开始模型评估...")
    # **保存完整的X和y，避免evaluate_model修改了作用域**
    X_full = X.copy() if hasattr(X, 'copy') else X
    y_full = y.copy() if hasattr(y, 'copy') else y
    groups_full = groups.copy() if groups is not None and hasattr(groups, 'copy') else groups
    keys_df_full = keys_df.copy() if isinstance(keys_df, pd.DataFrame) else keys_df
    
    results = []
    # RUL任务下移除容易产生极端外推的线性/SVR模型
    if str(target_type).lower() == 'rul':
        base = {k: v for k, v in base.items() if k in ['rf','xgb','torch_mlp','rf+ann']}
        print(f"RUL任务筛选后的模型: {list(base.keys())}")
    for name, model in base.items():
        print(f"正在评估模型: {name}")
        try:
            result = evaluate_model(
                name, model, X, y,
                cv_splits=cv_splits, cv_strategy=cv_strategy,
                selector=selector,
                groups=groups
            )
            results.append(result)
            print(f"模型 {name} 评估完成: RMSE={result['rmse']:.4f}")
        except Exception as e:
            print(f"模型 {name} 评估失败: {e}")

    # Stacking 构建处（稳定 rf+ann 输出）
    print("开始集成模型训练...")
    stack_base = {k: v for k, v in base.items() if k in ['xgb']}
    stack_base['rf+ann'] = RFPlusANNRegressor(cv=cv_splits, random_state=42, output_mode='correction')
    print(f"Stacking基模型: {list(stack_base.keys())}")
    stacking = build_stacking(stack_base, final_estimator=None, cv=cv_splits)

    print("正在评估Stacking模型...")
    results.append(
        evaluate_model(
            'stacking', stacking, X, y,
            cv_splits=cv_splits, cv_strategy=cv_strategy,
            selector=selector,
            groups=groups
        )
    )

    # 评估 Blending（核心=rf+ann）
    print("正在评估Blending模型（核心=rf+ann）...")
    blending_base = {k: v for k, v in base.items()}
    blending = BlendingRegressor(
        base_models=blending_base,
        cv=cv_splits,
        shrinkage=0.05,     # 降低收缩强度，避免权重被拉向均匀
        core_model='rf+ann',
        core_prior_weight=0.6,
        cv_repeats=1        # 减少重复K折次数，降低过度平滑
    )
    results.append(
        evaluate_model(
            'blending', blending, X, y,
            cv_splits=cv_splits, cv_strategy=cv_strategy,
            selector=selector,
            groups=groups
        )
    )

    if RUN_ONLY_STACKING:
        print("仅运行 Stacking，提前结束。")
        sys.exit(0)

    # 选择最优模型并生成后续产物
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k in ('name','rmse','mae','r2')} for r in results]).sort_values('rmse')
    best_result = min(results, key=lambda r: r['rmse'])
    best_name = best_result['name']
    y_pred_oof = np.asarray(best_result['y_pred'], dtype=float)
    y_true_used = np.asarray(best_result.get('y_true', y), dtype=float)

    # 用已拟合的选择器给 X_selected 加上列名
    sel_cols = list(getattr(selector, 'selected_features_', []))
    if len(sel_cols) != int(X_selected.shape[1]):
        sel_cols = [f'f_{i}' for i in range(X_selected.shape[1])]
    X_sel_full = pd.DataFrame(X_selected, columns=sel_cols)
    indices = np.asarray(best_result.get('indices', None)) if best_result.get('indices', None) is not None else None
    if indices is not None and len(indices) == len(y_pred_oof) and X_sel_full.shape[0] != len(y_pred_oof):
        try:
            X_sel_for_corr = X_sel_full.iloc[indices]
        except Exception:
            X_sel_for_corr = X_sel_full
    else:
        X_sel_for_corr = X_sel_full

    # 保存相关性与图表
    os.makedirs(art_dir, exist_ok=True)
    corr_pred = compute_feature_corr_df(X_sel_for_corr, y_pred_oof, method='spearman')
    corr_true = compute_feature_corr_df(X_sel_for_corr, y_true_used, method='spearman')
    corr_pred.to_csv(os.path.join(art_dir, 'feature_vs_pred_corr.csv'), index=False, encoding='utf-8-sig')
    corr_true.to_csv(os.path.join(art_dir, 'feature_vs_true_corr.csv'), index=False, encoding='utf-8-sig')

    print("Top-20 feature×prediction correlation:")
    print(corr_pred.head(20))
    df_res.to_csv(os.path.join(art_dir, 'model_compare.csv'), index=False, encoding='utf-8-sig')
    def _safe_wape(y_true, y_pred):
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]; yp = yp[mask]
        denom = float(np.sum(np.abs(yt))) + 1e-9
        if denom <= 1e-9:
            return float('nan')
        return float(np.sum(np.abs(yt - yp)) / denom)
    def _safe_mape(y_true, y_pred):
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        eps = 1e-3
        mask = np.isfinite(yt) & np.isfinite(yp) & (np.abs(yt) > eps)
        if not np.any(mask):
            return float('nan')
        yt = yt[mask]; yp = yp[mask]
        denom = np.maximum(np.abs(yt), eps)
        return float(np.mean(np.abs((yt - yp) / denom)))
    def _pinball(y_true, y_pred, tau):
        d = y_true - y_pred
        return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))
    wape = _safe_wape(y_true_used, y_pred_oof)
    mape = _safe_mape(y_true_used, y_pred_oof)
    pin_10 = _pinball(y_true_used, y_pred_oof, 0.1)
    pin_50 = _pinball(y_true_used, y_pred_oof, 0.5)
    pin_90 = _pinball(y_true_used, y_pred_oof, 0.9)
    neg_ratio = None
    try:
        if str(target_type).lower() == 'rul':
            yp = np.asarray(y_pred_oof, dtype=float)
            neg_ratio = float(np.mean(yp < 0.0))
            print(f"RUL负预测比例: {neg_ratio:.3f}")
    except Exception:
        pass
    pd.DataFrame([{
        'rmse': float(best_result['rmse']),
        'mae': float(best_result['mae']),
        'r2': float(best_result['r2']),
        'wape': float(wape),
        'mape': float(mape),
        'pinball_0.1': float(pin_10),
        'pinball_0.5': float(pin_50),
        'pinball_0.9': float(pin_90),
        'rul_negative_ratio': float(neg_ratio) if neg_ratio is not None else None
    }]).to_json(os.path.join(art_dir, 'final_metrics.json'), orient='records', force_ascii=False)
    try:
        if cv_strategy in ('group_holdout','groupholdout','holdout_groups'):
            pd.DataFrame([{
                'rmse': float(best_result['rmse']),
                'mae': float(best_result['mae']),
                'r2': float(best_result['r2']),
                'wape': float(wape),
                'mape': float(mape)
            }]).to_json(os.path.join(art_dir, 'group_holdout_metrics.json'), orient='records', force_ascii=False)
    except Exception:
        pass

    # 额外：若预测的是容量保持率(%)，输出预测跌破阈值的循环号（EOL 预测）
    try:
        if str(target_type).lower() == 'capacity_retention':
            thr = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
            idx = None
            if isinstance(keys_df, pd.DataFrame) and 'cycle' in keys_df.columns:
                cycles_all = keys_df['cycle'].values
                indices = best_result.get('indices', None)
                if indices is not None and len(y_pred_oof) == len(indices):
                    cycles_use = np.asarray(cycles_all)[indices]
                else:
                    cycles_use = np.asarray(cycles_all)[:len(y_pred_oof)]
                pos = np.where(y_pred_oof < thr)[0]
                idx = int(cycles_use[pos[0]]) if pos.size > 0 else None
            else:
                pos = np.where(y_pred_oof < thr)[0]
                idx = int(pos[0]) if pos.size > 0 else None
            with open(os.path.join(art_dir, 'predicted_eol.json'), 'w', encoding='utf-8') as f:
                import json
                json.dump({'threshold_pct': thr, 'predicted_eol_cycle': idx}, f, ensure_ascii=False)
    except Exception:
        pass

    # 分组评估（如有）
    if groups is not None and cv_strategy in ('groupkfold','group_kfold','group','logo','leaveonegroupout','leave_one_group_out','group_holdout','groupholdout','holdout_groups'):
        rows = []
        if len(y_pred_oof) == len(groups):
            uniq = pd.unique(groups)
            for gid in uniq:
                m = (groups == gid)
                if int(m.sum()) >= 2:
                    rows.append({
                        'group': str(gid),
                        'n': int(m.sum()),
                        'rmse': float(np.sqrt(mean_squared_error(y_true_used[m], y_pred_oof[m]))),
                        'mae': float(mean_absolute_error(y_true_used[m], y_pred_oof[m])),
                        'r2': float(r2_score(y_true_used[m], y_pred_oof[m]))
                    })
            uniq_count = int(len(pd.unique(groups)))
        else:
            try:
                subgroup = np.asarray(groups)[indices] if indices is not None else None
            except Exception:
                subgroup = None
            if subgroup is not None and len(subgroup) == len(y_pred_oof):
                uniq = pd.unique(subgroup)
                for gid in uniq:
                    m = (subgroup == gid)
                    if int(m.sum()) >= 2:
                        rows.append({
                            'group': str(gid),
                            'n': int(m.sum()),
                            'rmse': float(np.sqrt(mean_squared_error(y_true_used[m], y_pred_oof[m]))),
                            'mae': float(mean_absolute_error(y_true_used[m], y_pred_oof[m])),
                            'r2': float(r2_score(y_true_used[m], y_pred_oof[m]))
                        })
                uniq_count = int(len(pd.unique(subgroup)))
            else:
                uniq_count = None
        if rows:
            pd.DataFrame(rows).sort_values('rmse').to_csv(os.path.join(art_dir, 'group_cv_metrics.csv'), index=False, encoding='utf-8-sig')
        pd.DataFrame([{
            'cv_strategy': cv_strategy,
            'groups_unique': uniq_count,
            'rmse_all': float(np.sqrt(mean_squared_error(y_true_used, y_pred_oof))),
            'mae_all': float(mean_absolute_error(y_true_used, y_pred_oof)),
            'r2_all': float(r2_score(y_true_used, y_pred_oof)),
            'wape_all': float(_safe_wape(y_true_used, y_pred_oof)),
            'mape_all': float(_safe_mape(y_true_used, y_pred_oof)),
            'pinball_0.1_all': float(_pinball(y_true_used, y_pred_oof, 0.1)),
            'pinball_0.5_all': float(_pinball(y_true_used, y_pred_oof, 0.5)),
            'pinball_0.9_all': float(_pinball(y_true_used, y_pred_oof, 0.9))
        }]).to_json(os.path.join(art_dir, 'group_cv_overall.json'), orient='records', force_ascii=False)

    def _label_text(en, zh, lang):
        if lang == 'zh':
            return zh
        return en
    def _setup_font(lang):
        try:
            import matplotlib as mpl
            if lang == 'zh':
                font_name = os.environ.get('PLOT_FONT', '').strip()
                if font_name:
                    mpl.rcParams['font.sans-serif'] = [font_name]
                    mpl.rcParams['axes.unicode_minus'] = False
        except Exception:
            pass
    def _save_plot(fig, path_base):
        try:
            fig.savefig(path_base + '.png', dpi=300, bbox_inches='tight')
            fig.savefig(path_base + '.svg', bbox_inches='tight')
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
    label_lang = os.environ.get('LABEL_LANG', 'auto').lower()
    if label_lang not in ('zh','en'):
        label_lang = 'en'
    _setup_font(label_lang)
    try:
        import matplotlib.pyplot as plt
        lo = float(np.nanmin([np.nanmin(y), np.nanmin(y_pred_oof)]))
        hi = float(np.nanmax([np.nanmax(y), np.nanmax(y_pred_oof)]))
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.scatter(y_true_used, y_pred_oof, s=12, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], color='red', linewidth=1)
        ax.set_xlabel(_label_text('True', '真实值', label_lang))
        ax.set_ylabel(_label_text('Predicted', '预测值', label_lang))
        ax.set_title(_label_text('Parity Plot', '拟合对角图', label_lang))
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02, y=0.05)
        _save_plot(fig, os.path.join(art_dir, 'parity_plot'))
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        res = np.asarray(y_pred_oof, dtype=float) - np.asarray(y_true_used, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.hist(res, bins=30, color='#4C72B0', alpha=0.8)
        ax.set_xlabel(_label_text('Residual', '残差', label_lang))
        ax.set_ylabel(_label_text('Count', '频数', label_lang))
        ax.set_title(_label_text('Residual Histogram', '残差直方图', label_lang))
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02, y=0.05)
        _save_plot(fig, os.path.join(art_dir, 'residual_hist'))
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.scatter(y_pred_oof, res, s=12, alpha=0.7)
        ax.axhline(0.0, color='red', linewidth=1)
        ax.set_xlabel(_label_text('Predicted', '预测值', label_lang))
        ax.set_ylabel(_label_text('Residual', '残差', label_lang))
        ax.set_title(_label_text('Residual vs Predicted', '残差-预测散点', label_lang))
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02, y=0.05)
        _save_plot(fig, os.path.join(art_dir, 'residual_vs_pred'))
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        from scipy import stats
        res = np.asarray(y_pred_oof, dtype=float) - np.asarray(y_true_used, dtype=float)
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        stats.probplot(res, dist=stats.norm, plot=plt)
        plt.xlabel(_label_text('Theoretical Quantiles', '理论分位数', label_lang))
        plt.ylabel(_label_text('Ordered Residuals', '排序残差', label_lang))
        plt.title(_label_text('Residual Q-Q Plot', '残差Q-Q图', label_lang))
        _save_plot(fig, os.path.join(art_dir, 'qq_plot'))
    except Exception:
        pass

    # =================================================================
    # RUL任务输出：对每个测试集电池，用前60%重新训练模型，预测后40%
    # =================================================================
    if str(target_type).lower() == 'rul':
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.base import clone
            thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
            train_ratio = float(os.environ.get('TIMESERIES_TRAIN_RATIO', '0.6'))

            # 可选：加载 SOH 任务输出的容量保持率预测曲线，用于级联 SOH→RUL
            soh_curve_df = None
            try:
                # 优先从环境变量指定路径加载
                soh_curve_path = os.environ.get('SOH_CURVE_PATH', '').strip()
                if not soh_curve_path:
                    # 若未指定，则尝试在当前art_dir的父目录下查找标准SOH曲线文件
                    base_dir = os.path.dirname(art_dir)
                    cand1 = os.path.join(base_dir, 'soh', 'capacity_retention_curve.csv')
                    cand2 = os.path.join(base_dir, 'capacity_retention_curve.csv')
                    for _p in (cand1, cand2):
                        if os.path.isfile(_p):
                            soh_curve_path = _p
                            break
                if soh_curve_path and os.path.isfile(soh_curve_path):
                    print(f"尝试从SOH曲线加载级联特征: {soh_curve_path}")
                    soh_curve_df = pd.read_csv(soh_curve_path, encoding='utf-8-sig')
                    # 规范列名，期望包含: battery_id(or barcode)，cycle，capacity_retention_pct_stacking, capacity_retention_pct_blending
                    cols = {c.lower(): c for c in soh_curve_df.columns}
                    # battery_id / barcode 统一映射
                    id_col = None
                    for key in ['battery_id', 'barcode']:
                        if key in cols:
                            id_col = cols[key]
                            break
                    cyc_col = cols.get('cycle', None)
                    if id_col is None or cyc_col is None:
                        soh_curve_df = None
                    else:
                        soh_curve_df = soh_curve_df.copy()
                        soh_curve_df['__bat_id__'] = soh_curve_df[id_col].astype(str)
                        soh_curve_df['__cycle__'] = soh_curve_df[cyc_col].astype(float)
                else:
                    soh_curve_df = None
            except Exception:
                soh_curve_df = None
            
            print(f"\n{'='*60}")
            print(f"RUL任务：对每个测试集电池用前{int(train_ratio*100)}%重新训练，预测后{int((1-train_ratio)*100)}%")
            print(f"{'='*60}")
            
            # 获取最佳模型配置
            best_stacking = None
            best_blending = None
            for result in results:
                if result.get('name') == 'stacking':
                    best_stacking = result
                elif result.get('name') == 'blending':
                    best_blending = result
            
            if best_stacking is None or best_blending is None:
                raise Exception("缺少Stacking或Blending模型配置")
            
            # 识别测试集电池（后20%）
            if groups_full is None:
                raise Exception("groups_full为None")
            
            uniq_batteries = np.unique(np.asarray(groups_full).astype(object))
            rng = np.random.RandomState(int(os.environ.get('GROUP_HOLDOUT_SEED', '42')))
            order = rng.permutation(len(uniq_batteries))
            ratio = float(os.environ.get('GROUP_HOLDOUT_RATIO', '0.8'))
            k = max(1, int(np.floor(len(uniq_batteries) * ratio)))
            test_batteries = uniq_batteries[order[k:]]
            
            print(f"测试集电池数量: {len(test_batteries)}")
            print(f"测试集电池: {list(test_batteries)}")
            
            all_battery_data = []
            global_test_indices = []
            
            for bat_id in test_batteries:
                print(f"\n处理电池 {bat_id}...")
                
                # 提取该电池的所有数据
                bat_mask = (np.asarray(groups_full) == bat_id)
                bat_indices = np.where(bat_mask)[0]
                
                if len(bat_indices) == 0:
                    continue
                
                # 提取特征和目标
                X_bat = X_full.iloc[bat_indices].copy() if isinstance(X_full, pd.DataFrame) else X_full[bat_indices]
                
                # RUL任务：训练目标是容量保持率（不是RUL数值）
                # 从keys_df_full提取真实容量保持率作为目标
                if isinstance(keys_df_full, pd.DataFrame) and 'capacity_retention_pct' in keys_df_full.columns:
                    y_bat = keys_df_full.iloc[bat_indices]['capacity_retention_pct'].values
                    cap_ret_bat = y_bat.copy()
                else:
                    # 回退到原始RUL目标（但这不应该发生）
                    y_bat = y_full[bat_indices]
                    cap_ret_bat = None
                
                # 真实RUL值（用于后续对比和评估）
                # 从keys_df_full计算真实RUL，而不是使用y_full
                if isinstance(keys_df_full, pd.DataFrame) and 'capacity_retention_pct' in keys_df_full.columns:
                    cap_retention = keys_df_full.iloc[bat_indices]['capacity_retention_pct'].values
                    
                    # ========== 异常值处理和平滑(不删除数据点) ==========
                    from scipy.signal import savgol_filter
                    retention_cleaned = cap_retention.copy()
                    if len(cap_retention) >= 7:
                        rolling_median = pd.Series(cap_retention).rolling(window=7, min_periods=3, center=True).median()
                        rolling_std = pd.Series(cap_retention).rolling(window=7, min_periods=3, center=True).std()
                        outlier_mask = np.abs(cap_retention - rolling_median) > (3 * rolling_std)
                        outlier_mask = outlier_mask.fillna(False).values
                        
                        if outlier_mask.sum() > 0:
                            print(f"  {bat_id}: 检测到 {outlier_mask.sum()} 个异常值,用中位数替换")
                            retention_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
                    
                    # Savitzky-Golay平滑
                    if len(retention_cleaned) >= 11:
                        window = min(11, len(retention_cleaned) if len(retention_cleaned) % 2 == 1 else len(retention_cleaned) - 1)
                        smoothed = savgol_filter(retention_cleaned, window_length=window, polyorder=2, mode='nearest')
                        smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
                    elif len(retention_cleaned) >= 5:
                        window = min(5, len(retention_cleaned) if len(retention_cleaned) % 2 == 1 else len(retention_cleaned) - 1)
                        smoothed = savgol_filter(retention_cleaned, window_length=window, polyorder=2, mode='nearest')
                    else:
                        smoothed = retention_cleaned
                    
                    cap_retention = smoothed  # 使用平滑后的值
                    # ========================================
                    
                    # 计算真实RUL：找到容量≤80%的第一个点
                    rul_true_bat = np.zeros(len(cap_retention))
                    for i in range(len(cap_retention)):
                        # 从当前点开始找到第一个≤80%的点
                        remaining = 0
                        for j in range(i, len(cap_retention)):
                            if cap_retention[j] <= thr_pct:
                                remaining = j - i
                                break
                        rul_true_bat[i] = remaining
                else:
                    rul_true_bat = y_full[bat_indices]
                
                # 获取cycle
                if isinstance(keys_df_full, pd.DataFrame) and 'cycle' in keys_df_full.columns:
                    cycles_bat = keys_df_full.iloc[bat_indices]['cycle'].values
                else:
                    cycles_bat = np.arange(len(bat_indices))

                # 如果存在 SOH 预测曲线，则按 battery_id+cycle 对齐，提取 SOH 预测作为级联特征
                if soh_curve_df is not None:
                    try:
                        bat_str = str(bat_id)
                        df_soh_bat = soh_curve_df[soh_curve_df['__bat_id__'] == bat_str]
                        if not df_soh_bat.empty:
                            df_soh_bat = df_soh_bat.sort_values('__cycle__')
                            # 取 stacking / blending 预测的平均作为 soh_pred_pct
                            cand_cols = [c for c in df_soh_bat.columns if 'capacity_retention_pct_' in c.lower()]
                            if len(cand_cols) > 0:
                                df_tmp = df_soh_bat[['__cycle__'] + cand_cols].copy()
                                df_tmp['soh_pred_pct'] = df_tmp[cand_cols].astype(float).mean(axis=1)
                                # 将 cycles_bat 与 SOH 预测按cycle对齐
                                df_keys = pd.DataFrame({'__cycle__': np.asarray(cycles_bat, dtype=float)})
                                df_merged = df_keys.merge(df_tmp[['__cycle__', 'soh_pred_pct']], on='__cycle__', how='left')
                                soh_pred_vals = df_merged['soh_pred_pct'].values
                                if isinstance(X_bat, pd.DataFrame):
                                    X_bat = X_bat.copy()
                                    X_bat['soh_pred_pct'] = soh_pred_vals
                                else:
                                    import pandas as pd
                                    X_bat = pd.DataFrame(X_bat)
                                    X_bat['soh_pred_pct'] = soh_pred_vals
                    except Exception:
                        pass

                # 新增：为RUL任务加入归一化周期特征，帮助模型学习退化趋势
                try:
                    cycles_float = np.asarray(cycles_bat, dtype=float)
                    cyc_max = np.nanmax(cycles_float) if cycles_float.size > 0 else 0.0
                    if cyc_max > 0:
                        cycle_ratio = cycles_float / cyc_max
                    else:
                        cycle_ratio = np.zeros_like(cycles_float)
                    if isinstance(X_bat, pd.DataFrame):
                        X_bat = X_bat.copy()
                        X_bat['cycle_ratio'] = cycle_ratio
                    else:
                        import pandas as pd
                        X_bat = pd.DataFrame(X_bat)
                        X_bat['cycle_ratio'] = cycle_ratio
                except Exception:
                    pass

                # 新增：为RUL任务加入距离阈值80%的gap特征，帮助模型感知接近EOL的程度
                try:
                    y_bat_arr = np.asarray(y_bat, dtype=float)
                    gap_to_80 = y_bat_arr - thr_pct
                    if isinstance(X_bat, pd.DataFrame):
                        X_bat = X_bat.copy()
                        X_bat['gap_to_80_pct'] = gap_to_80
                    else:
                        import pandas as pd
                        X_bat = pd.DataFrame(X_bat)
                        X_bat['gap_to_80_pct'] = gap_to_80
                except Exception:
                    pass
                
                # 按cycle排序
                sort_order = np.argsort(cycles_bat)
                cycles_bat = cycles_bat[sort_order]
                y_bat = y_bat[sort_order]  # 容量保持率
                rul_true_bat = rul_true_bat[sort_order]  # 真实RUL
                X_bat = X_bat.iloc[sort_order].copy() if isinstance(X_bat, pd.DataFrame) else X_bat[sort_order]
                bat_indices_sorted = bat_indices[sort_order]
                if cap_ret_bat is not None:
                    cap_ret_bat = cap_ret_bat[sort_order]
                
                # 删除目标变量列
                if isinstance(X_bat, pd.DataFrame) and 'capacity_retention_pct' in X_bat.columns:
                    X_bat = X_bat.drop(columns=['capacity_retention_pct'])
                
                n_cycles = len(cycles_bat)
                n_train = int(n_cycles * train_ratio)
                
                if n_train >= n_cycles or n_train < 5:
                    print(f"  跳过电池{bat_id}：数据不足（共{n_cycles}个cycle）")
                    continue
                
                # 分割训练集和测试集
                X_train_bat = X_bat.iloc[:n_train] if isinstance(X_bat, pd.DataFrame) else X_bat[:n_train]
                y_train_bat = y_bat[:n_train]  # 前60%的容量保持率
                X_test_bat = X_bat.iloc[n_train:] if isinstance(X_bat, pd.DataFrame) else X_bat[n_train:]
                y_test_bat = y_bat[n_train:]  # 后40%的真实容量保持率（用于对比）

                # 对接近阈值区域(约80%)的样本进行重复采样，加大其在训练中的权重
                try:
                    y_train_arr = np.asarray(y_train_bat, dtype=float)
                    # 以阈值为中心，取一个区间，例如 70%~90%
                    lower = thr_pct - 10.0
                    upper = thr_pct + 10.0
                    focus_mask = np.isfinite(y_train_arr) & (y_train_arr >= lower) & (y_train_arr <= upper)
                    if focus_mask.sum() > 0:
                        # 重复采样倍数，可以根据需要调整
                        repeat = 3
                        if isinstance(X_train_bat, pd.DataFrame):
                            X_focus = X_train_bat.iloc[focus_mask].copy()
                            X_oversampled = pd.concat([X_train_bat] + [X_focus] * (repeat - 1), axis=0)
                        else:
                            X_focus = X_train_bat[focus_mask]
                            X_oversampled = np.concatenate([X_train_bat] + [X_focus] * (repeat - 1), axis=0)
                        y_focus = y_train_arr[focus_mask]
                        y_oversampled = np.concatenate([y_train_arr] + [y_focus] * (repeat - 1), axis=0)
                        X_train_bat, y_train_bat = X_oversampled, y_oversampled
                except Exception:
                    pass

                # 用前60%重新训练Stacking模型（目标：容量保持率）
                stacking_pipeline_bat = clone(best_stacking.get('pipeline'))
                stacking_pipeline_bat.fit(X_train_bat, y_train_bat)
                y_pred_cap_stacking = stacking_pipeline_bat.predict(X_test_bat)  # 预测容量保持率
                
                # 用前60%重新训练Blending模型（目标：容量保持率）
                blending_pipeline_bat = clone(best_blending.get('pipeline'))
                blending_pipeline_bat.fit(X_train_bat, y_train_bat)
                y_pred_cap_blending = blending_pipeline_bat.predict(X_test_bat)  # 预测容量保持率
                
                print(f"  电池{bat_id}: 前{n_train}个cycle训练，预测后{n_cycles-n_train}个cycle")
                
                # 构建数据：前60%真实值 + 后40%预测值
                # RUL任务：模型预测容量保持率，然后从预测曲线计算RUL
                
                # 计算RUL：从预测的容量保持率曲线找到第一个≤8 0%的点
                def calculate_rul_from_capacity(capacity_curve, current_idx, threshold=80.0):
                    """
                    从容量保持率曲线计算RUL
                    capacity_curve: 容量保持率数组
                    current_idx: 当前索引
                    threshold: 失效阈值
                    """
                    # 从当前点开始查找
                    for i in range(current_idx, len(capacity_curve)):
                        if capacity_curve[i] <= threshold:
                            return i - current_idx  # 剩余cycle数
                    # 如果没找到，说明还未达到阈值
                    return len(capacity_curve) - current_idx
                
                # 构建完整的容量保持率预测曲线（前60%真实 + 后40%预测）
                full_cap_stacking = np.concatenate([y_train_bat, y_pred_cap_stacking])
                full_cap_blending = np.concatenate([y_train_bat, y_pred_cap_blending])

                # 对预测段容量保持率进行温和的物理后处理：只抑制明显向上跳，不强制整体向80%收敛
                def _postprocess_capacity_curve(full_cap, y_true_full, n_train_local, thr):
                    full_cap = np.asarray(full_cap, dtype=float).copy()
                    n_total = full_cap.size
                    if n_total <= n_train_local:
                        return np.clip(full_cap, 0.0, 120.0)

                    # 仅对预测段做约束：如果某点相比前一点明显上跳，则截断到前一点
                    start = max(n_train_local, 1)
                    tol = 0.5  # 允许的小幅上浮（单位：百分比点）
                    for i in range(start, n_total):
                        if np.isfinite(full_cap[i - 1]):
                            if not np.isfinite(full_cap[i]):
                                full_cap[i] = full_cap[i - 1]
                            else:
                                if full_cap[i] > full_cap[i - 1] + tol:
                                    full_cap[i] = full_cap[i - 1]

                    full_cap = np.clip(full_cap, 0.0, 120.0)
                    return full_cap

                full_cap_stacking_pp = _postprocess_capacity_curve(full_cap_stacking, y_bat, n_train, thr_pct)
                full_cap_blending_pp = _postprocess_capacity_curve(full_cap_blending, y_bat, n_train, thr_pct)
                
                for i in range(n_cycles):
                    if i < n_train:
                        # 前60%：真实数据
                        row = {
                            'battery_id': str(bat_id),
                            'cycle': int(cycles_bat[i]),
                            'capacity_retention_pct_stacking': float(y_bat[i]),  # 真实值
                            'capacity_retention_pct_blending': float(y_bat[i]),  # 真实值
                            'capacity_retention_pct_true': float(y_bat[i]),
                            'rul_true': float(rul_true_bat[i]),
                            'is_prediction': False
                        }
                    else:
                        # 后40%：预测的容量保持率 + 计算出RUL
                        cap_pred_stacking = float(full_cap_stacking_pp[i])
                        cap_pred_blending = float(full_cap_blending_pp[i])
                        
                        # 从Stacking预测曲线计算RUL（基于后处理后的容量曲线）
                        rul_pred_stacking = calculate_rul_from_capacity(full_cap_stacking_pp, i, thr_pct)

                        # 从Blending预测曲线计算RUL（基于后处理后的容量曲线）
                        rul_pred_blending = calculate_rul_from_capacity(full_cap_blending_pp, i, thr_pct)
                        
                        row = {
                            'battery_id': str(bat_id),
                            'cycle': int(cycles_bat[i]),
                            'capacity_retention_pct_stacking': np.clip(cap_pred_stacking, 0.0, 120.0),
                            'capacity_retention_pct_blending': np.clip(cap_pred_blending, 0.0, 120.0),
                            'capacity_retention_pct_true': float(y_bat[i]),
                            'rul_pred_stacking': int(rul_pred_stacking),
                            'rul_pred_blending': int(rul_pred_blending),
                            'rul_true': float(rul_true_bat[i]),
                            'is_prediction': True
                        }
                    
                    all_battery_data.append(row)
                    global_test_indices.append(bat_indices_sorted[i])
            
            # 保存结果
            if len(all_battery_data) > 0:
                df_all = pd.DataFrame(all_battery_data)
                df_all.to_csv(os.path.join(art_dir, 'rul_curve.csv'), index=False, encoding='utf-8-sig')
                
                print(f"\n{'='*60}")
                print(f"✅ RUL曲线已保存: {os.path.join(art_dir, 'rul_curve.csv')}")
                print(f"✅ 数据行数: {len(df_all)}, 电池数量: {df_all['battery_id'].nunique()}")
                print(f"✅ 真实值点数: {(~df_all['is_prediction']).sum()}, 预测值点数: {df_all['is_prediction'].sum()}")
                print(f"{'='*60}")
                
                # 绘制容量保持率预测曲线（从 RUL预测反推）
                try:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    
                    for bat_id in df_all['battery_id'].unique():
                        df_bat = df_all[df_all['battery_id'] == bat_id].sort_values('cycle')
                        df_train = df_bat[~df_bat['is_prediction']]
                        df_test = df_bat[df_bat['is_prediction']]
                        
                        # 绘制完整的真实容量保持率曲线
                        valid_mask = df_bat['capacity_retention_pct_true'].notna()
                        if valid_mask.sum() > 0:
                            ax.plot(df_bat.loc[valid_mask, 'cycle'], 
                                   df_bat.loc[valid_mask, 'capacity_retention_pct_true'], 
                                   color='#4C72B0', linewidth=2.0, alpha=0.8, label='_nolegend_')
                        
                        # 绘制后40%的Stacking预测曲线（虚线）
                        if len(df_test) > 0:
                            valid_mask_stacking = df_test['capacity_retention_pct_stacking'].notna()
                            if valid_mask_stacking.sum() > 0:
                                ax.plot(df_test.loc[valid_mask_stacking, 'cycle'], 
                                       df_test.loc[valid_mask_stacking, 'capacity_retention_pct_stacking'], 
                                       color='#C44E52', linewidth=1.5, linestyle='--', alpha=0.7, label='_nolegend_')
                            
                            # 绘制后40%的Blending预测曲线（虚线）
                            valid_mask_blending = df_test['capacity_retention_pct_blending'].notna()
                            if valid_mask_blending.sum() > 0:
                                ax.plot(df_test.loc[valid_mask_blending, 'cycle'], 
                                       df_test.loc[valid_mask_blending, 'capacity_retention_pct_blending'], 
                                       color='#55A868', linewidth=1.5, linestyle='--', alpha=0.7, label='_nolegend_')
                    
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='#4C72B0', linewidth=2.5, label='True Capacity Retention'),
                        Line2D([0], [0], color='#C44E52', linewidth=2.5, linestyle='--', label='Stacking Prediction (from RUL)'),
                        Line2D([0], [0], color='#55A868', linewidth=2.5, linestyle='--', label='Blending Prediction (from RUL)'),
                        Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label=f'EOL Threshold ({thr_pct}%)')
                    ]
                    
                    ax.axhline(thr_pct, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
                    ax.set_xlabel('Cycle', fontsize=13, fontweight='bold')
                    ax.set_ylabel('Capacity Retention (%)', fontsize=13, fontweight='bold')
                    ax.set_ylim(0, 105)
                    ax.legend(handles=legend_elements, fontsize=12, loc='best')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.set_title('RUL Prediction: Capacity Degradation Trajectory', fontsize=14, fontweight='bold')
                    
                    fig.tight_layout()
                    fig.savefig(os.path.join(art_dir, 'rul_curve.png'), dpi=300, bbox_inches='tight')
                    fig.savefig(os.path.join(art_dir, 'rul_curve.pdf'), bbox_inches='tight')
                    plt.close(fig)
                    print("✅ RUL曲线图表已保存")
                except Exception as e:
                    print(f"RUL曲线绘制失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("❌ 未生成RUL数据")
                
        except Exception as e:
            print(f"❌ RUL输出失败: {e}")
            import traceback
            traceback.print_exc()

    try:
        import pandas as pd
        res = np.asarray(y_pred_oof, dtype=float) - np.asarray(y, dtype=float)
        pd.DataFrame({'true': np.asarray(y, dtype=float), 'pred': np.asarray(y_pred_oof, dtype=float)}).to_csv(
            os.path.join(art_dir, 'parity_plot_data.csv'), index=False, encoding='utf-8-sig')
        pd.DataFrame({'pred': np.asarray(y_pred_oof, dtype=float), 'true': np.asarray(y, dtype=float), 'residual': res}).to_csv(
            os.path.join(art_dir, 'residuals.csv'), index=False, encoding='utf-8-sig')
        try:
            from scipy import stats
            res_sorted = np.sort(res)
            n = len(res_sorted)
            probs = (np.arange(1, n + 1) - 0.5) / n
            theo = stats.norm.ppf(probs)
            pd.DataFrame({'theoretical': theo, 'ordered': res_sorted}).to_csv(
                os.path.join(art_dir, 'qq_plot_data.csv'), index=False, encoding='utf-8-sig')
        except Exception:
            pd.DataFrame({'ordered': np.sort(res)}).to_csv(
                os.path.join(art_dir, 'qq_plot_data.csv'), index=False, encoding='utf-8-sig')
    except Exception:
        pass

    # =================================================================
    # SOH任务输出：对测试集电池（后20%）的所有cycle用训练集重新训练模型预测
    # =================================================================
    if str(target_type).lower() == 'capacity_retention':
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.base import clone
            thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
            
            print(f"\n{'='*60}")
            print(f"SOH任务：用训练集（前80%电池）重新训练，预测测试集（后20%）")
            print(f"{'='*60}")
            
            # 获取最佳模型配置
            best_stacking = None
            best_blending = None
            for result in results:
                if result.get('name') == 'stacking':
                    best_stacking = result
                elif result.get('name') == 'blending':
                    best_blending = result
            
            if best_stacking is None or best_blending is None:
                raise Exception("缺少Stacking或Blending模型配置")
            
            # 识别训练集和测试集电池
            if groups_full is None:
                raise Exception("groups_full为None")
            
            uniq_batteries = np.unique(np.asarray(groups_full).astype(object))
            rng = np.random.RandomState(int(os.environ.get('GROUP_HOLDOUT_SEED', '42')))
            order = rng.permutation(len(uniq_batteries))
            ratio = float(os.environ.get('GROUP_HOLDOUT_RATIO', '0.8'))
            k = max(1, int(np.floor(len(uniq_batteries) * ratio)))
            
            train_batteries = uniq_batteries[order[:k]]  # 前80%
            test_batteries = uniq_batteries[order[k:]]   # 后20%
            
            print(f"训练集电池数量: {len(train_batteries)}")
            print(f"测试集电池数量: {len(test_batteries)}")
            print(f"测试集电池: {list(test_batteries)}")
            
            # 提取训练集数据
            train_mask = np.isin(groups_full, train_batteries)
            X_train = X_full.iloc[train_mask] if isinstance(X_full, pd.DataFrame) else X_full[train_mask]
            y_train = y_full[train_mask]
            
            # 删除目标变量列
            if isinstance(X_train, pd.DataFrame) and 'capacity_retention_pct' in X_train.columns:
                X_train = X_train.drop(columns=['capacity_retention_pct'])
            
            # 用训练集重新训练Stacking和Blending
            print(f"\n用训练集（{len(X_train)}条数据）重新训练模型...")
            
            stacking_retrained = clone(best_stacking.get('pipeline'))
            stacking_retrained.fit(X_train, y_train)
            print("✅ Stacking模型重新训练完成")
            
            blending_retrained = clone(best_blending.get('pipeline'))
            blending_retrained.fit(X_train, y_train)
            print("✅ Blending模型重新训练完成")
            
            # 对测试集电池进行预测
            all_battery_data = []
            global_test_indices = []
            
            for bat_id in test_batteries:
                print(f"\n预测电池 {bat_id}...")
                
                bat_mask = (np.asarray(groups_full) == bat_id)
                bat_indices = np.where(bat_mask)[0]
                
                if len(bat_indices) == 0:
                    continue
                
                # 提取该电池的特征
                X_bat = X_full.iloc[bat_indices].copy() if isinstance(X_full, pd.DataFrame) else X_full[bat_indices]
                
                # 真实容量保持率（从keys_df_full提取，而不是X_full）
                if isinstance(keys_df_full, pd.DataFrame) and 'capacity_retention_pct' in keys_df_full.columns:
                    y_true_bat = keys_df_full.iloc[bat_indices]['capacity_retention_pct'].values
                    
                    # ========== 异常值处理和平滑(不删除数据点) ==========
                    from scipy.signal import savgol_filter
                    retention_cleaned = y_true_bat.copy()
                    if len(y_true_bat) >= 7:
                        rolling_median = pd.Series(y_true_bat).rolling(window=7, min_periods=3, center=True).median()
                        rolling_std = pd.Series(y_true_bat).rolling(window=7, min_periods=3, center=True).std()
                        outlier_mask = np.abs(y_true_bat - rolling_median) > (3 * rolling_std)
                        outlier_mask = outlier_mask.fillna(False).values
                        
                        if outlier_mask.sum() > 0:
                            print(f"  {bat_id}: 检测到 {outlier_mask.sum()} 个异常值,用中位数替换")
                            retention_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
                    
                    # Savitzky-Golay平滑
                    if len(retention_cleaned) >= 11:
                        window = min(11, len(retention_cleaned) if len(retention_cleaned) % 2 == 1 else len(retention_cleaned) - 1)
                        smoothed = savgol_filter(retention_cleaned, window_length=window, polyorder=2, mode='nearest')
                        smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
                    elif len(retention_cleaned) >= 5:
                        window = min(5, len(retention_cleaned) if len(retention_cleaned) % 2 == 1 else len(retention_cleaned) - 1)
                        smoothed = savgol_filter(retention_cleaned, window_length=window, polyorder=2, mode='nearest')
                    else:
                        smoothed = retention_cleaned
                    
                    y_true_bat = smoothed  # 使用平滑后的值
                    # ========================================
                else:
                    y_true_bat = y_full[bat_indices]
                
                # 获取cycle
                if isinstance(keys_df_full, pd.DataFrame) and 'cycle' in keys_df_full.columns:
                    cycles_bat = keys_df_full.iloc[bat_indices]['cycle'].values
                else:
                    cycles_bat = np.arange(len(bat_indices))
                
                # 排序
                sort_order = np.argsort(cycles_bat)
                cycles_bat = cycles_bat[sort_order]
                y_true_bat = y_true_bat[sort_order]
                X_bat = X_bat.iloc[sort_order].copy() if isinstance(X_bat, pd.DataFrame) else X_bat[sort_order]
                bat_indices_sorted = bat_indices[sort_order]
                
                # 删除目标变量列
                if isinstance(X_bat, pd.DataFrame) and 'capacity_retention_pct' in X_bat.columns:
                    X_bat = X_bat.drop(columns=['capacity_retention_pct'])
                
                # 用重新训练的模型预测所有cycle
                y_pred_stacking = stacking_retrained.predict(X_bat)
                y_pred_blending = blending_retrained.predict(X_bat)
                
                print(f"  电池{bat_id}: 预测{len(cycles_bat)}个cycle")
                
                # 构建数据
                for i in range(len(cycles_bat)):
                    row = {
                        'battery_id': str(bat_id),
                        'cycle': cycles_bat[i],
                        'capacity_retention_pct_stacking': np.clip(float(y_pred_stacking[i]), 0.0, 120.0),
                        'capacity_retention_pct_blending': np.clip(float(y_pred_blending[i]), 0.0, 120.0),
                        'capacity_retention_pct_true': float(y_true_bat[i])
                    }
                    all_battery_data.append(row)
                    global_test_indices.append(bat_indices_sorted[i])
            
            # 保存结果
            if len(all_battery_data) > 0:
                df_all = pd.DataFrame(all_battery_data)

                # 对预测容量保持率曲线进行平滑处理（按电池分组，顺序与真实值一致）
                try:
                    from scipy.signal import savgol_filter

                    def _smooth_capacity_series(arr):
                        arr = np.asarray(arr, dtype=float)
                        if arr.size >= 11:
                            window = min(11, arr.size if arr.size % 2 == 1 else arr.size - 1)
                            sm = savgol_filter(arr, window_length=window, polyorder=2, mode='nearest')
                            sm = pd.Series(sm).rolling(window=5, min_periods=1, center=True).mean().values
                        elif arr.size >= 5:
                            window = min(5, arr.size if arr.size % 2 == 1 else arr.size - 1)
                            sm = savgol_filter(arr, window_length=window, polyorder=2, mode='nearest')
                        else:
                            sm = arr
                        return sm

                    for _bid, g in df_all.groupby('battery_id', sort=False):
                        g_sorted = g.sort_values('cycle')
                        for col in ['capacity_retention_pct_stacking', 'capacity_retention_pct_blending']:
                            vals = g_sorted[col].values.astype(float)
                            mask = np.isfinite(vals)
                            if mask.sum() >= 5:
                                vals_sm = _smooth_capacity_series(vals[mask])
                                vals[mask] = vals_sm
                                df_all.loc[g_sorted.index, col] = vals
                except Exception:
                    pass

                df_all.to_csv(os.path.join(art_dir, 'capacity_retention_curve.csv'), index=False, encoding='utf-8-sig')

                print(f"\n{'='*60}")
                print(f"✅ SOH曲线已保存: {os.path.join(art_dir, 'capacity_retention_curve.csv')}")
                print(f"✅ 数据行数: {len(df_all)}, 电池数量: {df_all['battery_id'].nunique()}")
                print(f"{'='*60}")
                
                # 绘制曲线
                try:
                    fig, ax = plt.subplots(figsize=(14, 7))
                    
                    for bat_id in df_all['battery_id'].unique():
                        df_bat = df_all[df_all['battery_id'] == bat_id].sort_values('cycle')
                        
                        ax.plot(df_bat['cycle'], df_bat['capacity_retention_pct_stacking'], 
                               color='#C44E52', linewidth=1.5, alpha=0.6, label='_nolegend_')
                        ax.plot(df_bat['cycle'], df_bat['capacity_retention_pct_blending'], 
                               color='#55A868', linewidth=1.5, alpha=0.6, label='_nolegend_')
                        ax.plot(df_bat['cycle'], df_bat['capacity_retention_pct_true'], 
                               color='#4C72B0', linewidth=1.5, alpha=0.6, label='_nolegend_')
                    
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], color='#4C72B0', linewidth=2.5, label='True Values'),
                        Line2D([0], [0], color='#C44E52', linewidth=2.5, label='Stacking'),
                        Line2D([0], [0], color='#55A868', linewidth=2.5, label='Blending'),
                        Line2D([0], [0], color='red', linestyle='--', linewidth=2.5, label=f'EOL ({thr_pct}%)')
                    ]
                    
                    ax.axhline(thr_pct, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
                    ax.set_xlabel('Cycle', fontsize=13, fontweight='bold')
                    ax.set_ylabel('Capacity Retention (%)', fontsize=13, fontweight='bold')
                    ax.set_ylim(0, 105)
                    ax.legend(handles=legend_elements, fontsize=12, loc='best')
                    ax.grid(True, alpha=0.3, linestyle='--')
                    fig.tight_layout()
                    fig.savefig(os.path.join(art_dir, 'capacity_retention_curve.png'), dpi=300, bbox_inches='tight')
                    fig.savefig(os.path.join(art_dir, 'capacity_retention_curve.pdf'), bbox_inches='tight')
                    plt.close(fig)
                    print("✅ SOH曲线图表已保存")
                except Exception as e:
                    print(f"SOH曲线绘制失败: {e}")
            else:
                print("❌ 未生成SOH数据")
                
        except Exception as e:
            print(f"❌ SOH输出失败: {e}")
            import traceback
            traceback.print_exc()

    # 按电池ID分别导出（优先使用“条码”，否则使用分组文件名）
    try:
        import pandas as pd
        os.makedirs(os.path.join(art_dir, 'by_battery'), exist_ok=True)
        def _sanitize(s):
            try:
                return ''.join(c if c.isalnum() or c in ('-','_','.') else '_' for c in str(s))
            except Exception:
                return 'unknown'
        # **使用完整的keys_df_full，避免keys_df被修改**
        # 对齐到预测子集的 keys
        # **使用global_test_indices，这是从RUL/SOH处理时保存的测试集索引**
        if isinstance(keys_df_full, pd.DataFrame):
            keys_pred = keys_df_full.iloc[global_test_indices] if ('global_test_indices' in locals() and global_test_indices is not None) else keys_df_full
        else:
            keys_pred = pd.DataFrame({'cycle': np.arange(len(y_pred_oof))})
        # 选择ID列
        if 'barcode' in keys_pred.columns:
            id_values = keys_pred['barcode'].astype(str).values
            id_name = 'barcode'
        else:
            # 回退到文件分组（目录模式）
            grp_pred = None
            try:
                # **使用完整的groups_full和global_test_indices**
                grp_pred = np.asarray(groups_full)[global_test_indices] if ('global_test_indices' in locals() and global_test_indices is not None and groups_full is not None) else (np.asarray(groups_full) if groups_full is not None else None)
            except Exception:
                grp_pred = None
            if grp_pred is None or len(grp_pred) != len(y_pred_oof):
                id_values = np.array(['group'], dtype=object)
            else:
                id_values = grp_pred.astype(str)
            id_name = 'group'
        # 周期列
        if 'cycle' in keys_pred.columns:
            cyc_values = np.asarray(keys_pred['cycle'].values, dtype=float)
        else:
            cyc_values = np.arange(len(y_pred_oof))
        cap_true_all = None
        rul_true_all = None
        if str(target_type).lower() == 'capacity_retention':
            cap_true_all = y_true_used
        elif str(target_type).lower() == 'rul':
            rul_true_all = y_true_used
        elif isinstance(X_full, pd.DataFrame) and 'capacity_retention_pct' in X_full.columns:
            try:
                # **使用完整的X_full**
                cap_true_all = X_full['capacity_retention_pct'].values
            except Exception:
                cap_true_all = None
        # 汇总CSV
        rows_all = []
        thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
        summary_rows = []
        uniq_ids = pd.unique(id_values) if len(id_values) == len(y_pred_oof) else np.array(['all'])
        forecast_rows = []
        # 不再生成by_battery单个电池的图像文件
        # 所有电池数据已在rul_curve.csv和capacity_retention_curve.csv中
        # 包含：条码, cycle, Stacking预测, Blending预测, 真实值
        print(f"\n所有电池数据已汇总到：")
        if str(target_type).lower() == 'rul':
            print(f"  - {os.path.join(art_dir, 'rul_curve.csv')}")
        else:
            print(f"  - {os.path.join(art_dir, 'capacity_retention_curve.csv')}")
        print(f"包含所有电池的Stacking和Blending预测结果")

        # 叠加图：前段训练/后段预测容量(Ah)（stacking/blending）
        try:
            env_list = os.environ.get('OVERLAY_MODELS', '').strip()
            models_overlay = [s.strip() for s in env_list.split(',') if s.strip()] if env_list else ['ridge','svr','xgb','stacking','blending']
            thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80'))
            ratio = float(os.environ.get('FORECAST_TRAIN_RATIO', '0.4'))
            ratio = 0.4 if (ratio <= 0.0 or ratio >= 1.0) else ratio
            for uid in uniq_ids:
                m_all = (id_values == uid) if len(id_values) == len(y_pred_oof) else np.ones(len(y_pred_oof), dtype=bool)
                idx_all = np.where(m_all)[0]
                if idx_all.size < 5:
                    continue
                cyc_all = cyc_values[idx_all] if len(cyc_values) == len(y_pred_oof) else np.arange(idx_all.size)
                order = np.argsort(cyc_all, kind='mergesort')
                idx_sorted = idx_all[order]
                n = idx_sorted.size
                k = max(1, int(np.floor(n * ratio)))
                train_idx_b = idx_sorted[:k]
                test_idx_b = idx_sorted[k:]
                train_end_cycle = float(cyc_all[order][k-1]) if k >= 1 else float(cyc_all[order][0])
                # 基准容量（该电池首循环放电容量Ah），按 cycle 排序后取首条
                base_ah = np.nan
                try:
                    # **使用完整的X_full**
                    if isinstance(X_full, pd.DataFrame) and {'discharge_capacity_ah','cycle'}.issubset(set(X_full.columns)):
                        xmask = m_all if len(X_full) == len(y_pred_oof) else np.ones(len(X_full), dtype=bool)
                        xb = X_full.loc[xmask, ['discharge_capacity_ah','cycle']].dropna()
                        if xb.shape[0]:
                            base_ah = float(xb.sort_values('cycle').iloc[0]['discharge_capacity_ah'])
                    if not np.isfinite(base_ah) and isinstance(X_full, pd.DataFrame) and 'discharge_capacity_ah' in X_full.columns:
                        xb2 = X_full['discharge_capacity_ah'].dropna()
                        if xb2.shape[0]:
                            base_ah = float(xb2.iloc[0])
                    # 读取原始CSV作为可靠基准
                    import os
                    data_dir = os.environ.get('BATTERY_XLSX_DIR', '').strip()
                    if (not np.isfinite(base_ah)) and data_dir and isinstance(uid, (str, bytes)):
                        raw_path = os.path.join(data_dir, str(uid))
                        if os.path.isfile(raw_path):
                            from build_dataset import read_battery_csv
                            s1_raw, _, _ = read_battery_csv(raw_path)
                            if ('cycle' in s1_raw.columns) and ('discharge_capacity_ah' in s1_raw.columns):
                                dfr = s1_raw[['cycle','discharge_capacity_ah']].dropna().groupby('cycle', sort=True)['discharge_capacity_ah'].mean()
                                if dfr.shape[0]:
                                    base_ah = float(dfr.iloc[0])
                except Exception:
                    pass
                if not np.isfinite(base_ah):
                    base_ah = 1.0
                # 真值保持率%（全段）
                if cap_true_all is not None:
                    true_pct_all = np.asarray(cap_true_all, dtype=float)[idx_sorted]
                elif isinstance(X_full, pd.DataFrame) and 'capacity_retention_pct' in X_full.columns:
                    # **使用完整的X_full**
                    true_pct_all = np.asarray(X_full['capacity_retention_pct'].values, dtype=float)[idx_sorted]
                else:
                    continue
                cap_true_ah_all = base_ah * np.clip(true_pct_all, 0.0, 100.0) / 100.0
                eol_ah = 0.8 * base_ah
                df_overlay = pd.DataFrame({'cycle': cyc_all[order], 'cap_true_ah': cap_true_ah_all})
                try:
                    import os
                    data_dir = os.environ.get('BATTERY_XLSX_DIR', '').strip()
                    if data_dir and isinstance(uid, (str, bytes)):
                        raw_path = os.path.join(data_dir, str(uid))
                        if os.path.isfile(raw_path):
                            from build_dataset import read_battery_csv
                            s1_raw, _, _ = read_battery_csv(raw_path)
                            if ('cycle' in s1_raw.columns) and ('discharge_capacity_ah' in s1_raw.columns):
                                dfr = s1_raw[['cycle','discharge_capacity_ah']].dropna().groupby('cycle', sort=True)['discharge_capacity_ah'].mean()
                                cyc_raw = dfr.index.values
                                cap_raw = dfr.values
                                pos = np.searchsorted(df_overlay['cycle'].values, cyc_raw)
                                for pi, val in zip(pos, cap_raw):
                                    if 0 <= pi < df_overlay.shape[0]:
                                        df_overlay.at[pi, 'cap_true_ah'] = float(val)
                except Exception:
                    pass
                m_true = np.isfinite(df_overlay['cap_true_ah'].values) & (df_overlay['cap_true_ah'].values > 0.1)
                df_overlay = df_overlay[m_true]
                pred_eol_map = {}
                for mname in models_overlay:
                    r = None
                    for item in results:
                        if item.get('name') == mname:
                            r = item; break
                    if r is None:
                        continue
                    ypred = np.asarray(r.get('y_pred'), dtype=float)
                    indices_m = r.get('indices', None)
                    if indices_m is None:
                        continue
                    # 提取该电池测试段预测并映射到Ah
                    mask_m_test = np.isin(indices_m, test_idx_b)
                    idx_m_test = np.asarray(indices_m)[mask_m_test]
                    if idx_m_test.size == 0:
                        continue
                    # 预测序按电池测试索引顺序对齐
                    ypred_test = ypred[mask_m_test]
                    cycles_test = cyc_values[idx_m_test] if len(cyc_values) == len(y_pred_oof) else np.arange(idx_m_test.size)
                    cap_pred_ah = base_ah * np.clip(ypred_test, 0.0, 100.0) / 100.0
                    df_overlay[f'cap_pred_ah_{mname}'] = np.nan
                    # 写入对应位置
                    pos_in_sorted = np.searchsorted(df_overlay['cycle'].values, cycles_test)
                    for pi, val in zip(pos_in_sorted, cap_pred_ah):
                        if 0 <= pi < df_overlay.shape[0]:
                            df_overlay.at[pi, f'cap_pred_ah_{mname}'] = val
                    # 预测EOL（测试段首次 < eol_ah）
                    try:
                        series_pred = df_overlay[f'cap_pred_ah_{mname}'].values
                        valid_mask = ~np.isnan(series_pred)
                        idx_first = np.where((series_pred[valid_mask] < eol_ah))[0]
                        if idx_first.size > 0:
                            cyc_vals_valid = df_overlay['cycle'].values[valid_mask]
                            pred_eol = float(cyc_vals_valid[idx_first[0]])
                        else:
                            pred_eol = np.nan
                        pred_eol_map[mname] = pred_eol
                    except Exception:
                        pred_eol_map[mname] = np.nan
                # 真实EOL（全段首次 < eol_ah）
                try:
                    idx_true_eol = np.where(df_overlay['cap_true_ah'].values < eol_ah)[0]
                    true_eol = float(df_overlay['cycle'].values[idx_true_eol[0]]) if idx_true_eol.size > 0 else np.nan
                except Exception:
                    true_eol = np.nan
                # 画图
                try:
                    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
                    ax.plot(df_overlay['cycle'], df_overlay['cap_true_ah'], color='#4C4C4C', linewidth=1.7, label=_label_text('Actual', '真实容量Ah', label_lang))
                    palette = ['#C44E52','#55A868','#4C72B0','#8172B3','#CCB974','#64B5CD','#8C564B','#E377C2']
                    ci = 0
                    for mname in models_overlay:
                        coln = f'cap_pred_ah_{mname}'
                        if coln in df_overlay.columns:
                            ax.plot(df_overlay['cycle'], df_overlay[coln], linewidth=1.6, linestyle='--', label=mname, color=palette[ci % len(palette)])
                            ci += 1
                    ax.axvline(train_end_cycle, color='red', linestyle='-', linewidth=1)
                    ax.axhline(eol_ah, color='red', linestyle='--', linewidth=1)
                    for mname, peol in pred_eol_map.items():
                        if np.isfinite(peol):
                            ax.axvline(peol, color='gray', linestyle=':', linewidth=1)
                    ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
                    ax.set_ylabel(_label_text('Capacity (Ah)', '容量(Ah)', label_lang))
                    ax.grid(True, alpha=0.3); ax.legend(); ax.margins(x=0.02, y=0.05)
                    base = os.path.join(art_dir, 'by_battery', f'{id_name}_{_sanitize(uid)}_forecast_capacity_overlay')
                    _save_plot(fig, base)
                except Exception:
                    pass
                # 导出CSV与汇总
                try:
                    df_overlay.to_csv(os.path.join(art_dir, 'by_battery', f'{id_name}_{_sanitize(uid)}_forecast_capacity_overlay.csv'), index=False, encoding='utf-8-sig')
                except Exception:
                    pass
                row = {
                    'id': str(uid),
                    'train_end_cycle': train_end_cycle,
                    'base_capacity_ah': base_ah,
                    'true_eol_cycle': true_eol,
                    'TRUL': (true_eol - train_end_cycle) if np.isfinite(true_eol) else np.nan,
                    'predicted_eol_cycle_stacking': pred_eol_map.get('stacking', np.nan),
                    'PRUL_stacking': (pred_eol_map.get('stacking', np.nan) - train_end_cycle) if np.isfinite(pred_eol_map.get('stacking', np.nan)) else np.nan,
                    'predicted_eol_cycle_blending': pred_eol_map.get('blending', np.nan),
                    'PRUL_blending': (pred_eol_map.get('blending', np.nan) - train_end_cycle) if np.isfinite(pred_eol_map.get('blending', np.nan)) else np.nan,
                }
                forecast_rows.append(row)
            if forecast_rows:
                try:
                    pd.DataFrame(forecast_rows).to_csv(os.path.join(art_dir, 'by_battery_forecast_summary.csv'), index=False, encoding='utf-8-sig')
                except Exception:
                    pass
        except Exception:
            pass

        # by_battery文件已由前面的RUL/SOH汇总输出生成，不需要重复生成
        # 前面的代码已经生成了包含所有电池完整数据的CSV文件：
        # - rul_curve.csv (RUL任务)
        # - capacity_retention_curve.csv (SOH任务)
        # 这些文件包含所有电池的cycle、Stacking预测、Blending预测和真实值
        print("\nby_battery单独文件生成已跳过，请使用汇总CSV文件（包含所有电池数据）")
        print(f"  - RUL任务: {os.path.join(art_dir, 'rul_curve.csv')}")
        print(f"  - SOH任务: {os.path.join(art_dir, 'capacity_retention_curve.csv')}")
    except Exception:
        pass

    # 汇总容量衰减曲线 + 不确定性带（Blending / Stacking）
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        def _summary_curve(result_name, save_base):
            r = None
            for item in results:
                if item.get('name') == result_name:
                    r = item
                    break
            if r is None:
                return False
            yp = np.asarray(r.get('y_pred'), dtype=float)
            # 获取容量保持率真值；严格来源与兜底
            true_src = None
            if str(target_type).lower() == 'capacity_retention':
                yt = np.asarray(r.get('y_true', y_true_used), dtype=float)
                true_src = 'capacity_retention_target'
            elif isinstance(X_full, pd.DataFrame) and 'capacity_retention_pct' in X_full.columns:
                try:
                    # **使用完整的X_full**
                    yt = np.asarray(X_full['capacity_retention_pct'].values, dtype=float)
                    yt = yt[:len(yp)]
                    true_src = 'X_capacity_retention_pct'
                except Exception:
                    return False
            else:
                try:
                    # 兜底：重建保持率数据集（英文列），仅取真值
                    if data_mode == 'file' and os.path.isfile(xlsx_path):
                        _, y_cap, keys_cap = build_dataset_from_excel(xlsx_path, target_type='capacity_retention', capacity_threshold_pct=threshold_pct, return_keys=True)
                        yt = np.asarray(y_cap, dtype=float)[:len(yp)]
                        true_src = 'fallback_from_capacity_ah'
                    elif data_mode == 'dir' and os.path.isdir(xlsx_dir):
                        _, y_cap, _, keys_cap = build_dataset_from_directory(xlsx_dir, target_type='capacity_retention', capacity_threshold_pct=threshold_pct, return_keys=True)
                        yt = np.asarray(y_cap, dtype=float)[:len(yp)]
                        true_src = 'fallback_from_capacity_ah'
                    else:
                        return False
                except Exception:
                    return False
            idx = r.get('indices', None)
            kdf = None
            if isinstance(keys_df, pd.DataFrame):
                try:
                    kdf = keys_df.iloc[idx] if (idx is not None) else keys_df
                except Exception:
                    kdf = keys_df
            cycles_local = None
            if isinstance(kdf, pd.DataFrame) and 'cycle' in kdf.columns:
                cycles_local = kdf['cycle'].values
            else:
                cycles_local = np.arange(len(yp))
            # 构造表并聚合
            df = pd.DataFrame({'cycle': cycles_local, 'pred': yp, 'true': yt})
            # 数值安全
            df = df.replace([np.inf, -np.inf], np.nan)
            df['pred'] = np.clip(df['pred'].fillna(df['pred'].median()), 0.0, 100.0)
            df['true'] = np.clip(df['true'].fillna(df['true'].median()), 0.0, 100.0)
            # 调试：若真值近乎常数，输出预览并警告
            try:
                import numpy as np
                var_true = float(np.nanvar(df['true'].values))
                if var_true < 1e-6:
                    print(f"Warning: capacity_true appears constant (var={var_true:.6e}); source={true_src}")
                # 导出预览
                preview = df.copy()
                preview['source'] = true_src
                try:
                    preview.head(200).to_csv(os.path.join(art_dir, 'debug_true_retention_preview.csv'), index=False, encoding='utf-8-sig')
                except Exception:
                    pass
                try:
                    import json as _json
                    with open(os.path.join(art_dir, 'column_usage.json'), 'w', encoding='utf-8') as f:
                        _json.dump({'true_source': true_src, 'n_rows_debug': int(min(200, len(preview)))}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            except Exception:
                pass
            grp = df.groupby('cycle')
            out = pd.DataFrame({
                'cycle': grp.size().index,
                'pred_mean': grp['pred'].mean().values,
                'pred_median': grp['pred'].median().values,
                'pred_q10': grp['pred'].quantile(0.10).values,
                'pred_q90': grp['pred'].quantile(0.90).values,
                'true_mean': grp['true'].mean().values,
                'true_median': grp['true'].median().values,
                'true_q10': grp['true'].quantile(0.10).values,
                'true_q90': grp['true'].quantile(0.90).values,
            }).sort_values('cycle')
            # 保存CSV
            out.to_csv(save_base + '.csv', index=False, encoding='utf-8-sig')
            # 画图
            fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
            ax.plot(out['cycle'], out['true_mean'], color='#4C72B0', linewidth=1.6, label=_label_text('True mean', '真实均值', label_lang))
            ax.plot(out['cycle'], out['pred_mean'], color='#55A868', linewidth=1.6, label=_label_text(f'{result_name} mean', f'{result_name} 均值', label_lang))
            ax.fill_between(out['cycle'], out['pred_q10'], out['pred_q90'], color='#55A868', alpha=0.15, label=_label_text(f'{result_name} 10-90%', f'{result_name} 10-90%', label_lang))
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('Capacity Retention (%)', '容量保持率(%)', label_lang))
            try:
                y_top = float(max(100.0, 1.05 * max(
                    np.nanmax(out['true_mean']),
                    np.nanmax(out['pred_mean']),
                    np.nanmax(out['pred_q90'])
                )))
            except Exception:
                y_top = 100.0
            ax.set_ylim(0, y_top)
            try:
                ax.axhline(80.0, color='gray', linestyle=':', linewidth=1)
            except Exception:
                pass
            ax.grid(True, alpha=0.3)
            ax.margins(x=0.02, y=0.05)
            ax.legend()
            _save_plot(fig, save_base)
            return True
        if str(os.environ.get('ENABLE_MEAN_SUMMARY','0')).strip() == '1':
            _summary_curve('blending', os.path.join(art_dir, 'model_capacity_curve_mean_blending'))
            _summary_curve('stacking', os.path.join(art_dir, 'model_capacity_curve_mean_stacking'))
    except Exception:
        pass

    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        def _summary_curve_rul(result_name, save_base):
            r = None
            for item in results:
                if item.get('name') == result_name:
                    r = item
                    break
            if r is None:
                return False
            yp = np.asarray(r.get('y_pred'), dtype=float)
            yt = np.asarray(r.get('y_true', y_true_used), dtype=float)
            idx = r.get('indices', None)
            kdf = None
            if isinstance(keys_df, pd.DataFrame):
                try:
                    kdf = keys_df.iloc[idx] if (idx is not None) else keys_df
                except Exception:
                    kdf = keys_df
            cycles_local = None
            if isinstance(kdf, pd.DataFrame) and 'cycle' in kdf.columns:
                cycles_local = kdf['cycle'].values
            else:
                cycles_local = np.arange(len(yp))
            df = pd.DataFrame({'cycle': cycles_local, 'pred': yp, 'true': yt}).replace([np.inf, -np.inf], np.nan)
            df['pred'] = df['pred'].fillna(df['pred'].median())
            df['true'] = df['true'].fillna(df['true'].median())
            grp = df.groupby('cycle')
            out = pd.DataFrame({
                'cycle': grp.size().index,
                'pred_mean': grp['pred'].mean().values,
                'pred_median': grp['pred'].median().values,
                'pred_q10': grp['pred'].quantile(0.10).values,
                'pred_q90': grp['pred'].quantile(0.90).values,
                'true_mean': grp['true'].mean().values,
                'true_median': grp['true'].median().values,
                'true_q10': grp['true'].quantile(0.10).values,
                'true_q90': grp['true'].quantile(0.90).values,
            }).sort_values('cycle')
            out.to_csv(save_base + '.csv', index=False, encoding='utf-8-sig')
            fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
            ax.plot(out['cycle'], out['true_mean'], color='#4C72B0', linewidth=1.6, label=_label_text('True mean', '真实均值', label_lang))
            ax.plot(out['cycle'], out['pred_mean'], color='#55A868', linewidth=1.6, label=_label_text(f'{result_name} mean', f'{result_name} 均值', label_lang))
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('RUL (cycles)', '剩余寿命(圈数)', label_lang))
            try:
                y_top = float(1.05 * np.nanmax([np.nanmax(out['true_mean']), np.nanmax(out['pred_mean']), np.nanmax(out['pred_q90'])]))
            except Exception:
                y_top = float(np.nanmax(out['pred_mean'])) if out.shape[0] else 1.0
            ax.set_ylim(0.0, y_top)
            try:
                ax.axhline(0.0, color='gray', linestyle=':', linewidth=1)
            except Exception:
                pass
            ax.grid(True, alpha=0.3)
            ax.margins(x=0.02, y=0.05)
            ax.legend()
            _save_plot(fig, save_base)
            return True
        if str(target_type).lower() == 'rul' and str(os.environ.get('ENABLE_MEAN_SUMMARY','0')).strip() == '1':
            _summary_curve_rul('blending', os.path.join(art_dir, 'model_rul_curve_mean_blending'))
            _summary_curve_rul('stacking', os.path.join(art_dir, 'model_rul_curve_mean_stacking'))
    except Exception:
        pass

    # 代表性电池对比（Blending vs Stacking）
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        # 构造 per-battery 误差用于选择“误差中位”
        def _get_result(name):
            for item in results:
                if item.get('name') == name:
                    return item
            return None
        r_bl = _get_result('blending')
        r_st = _get_result('stacking')
        if r_bl is None or r_st is None:
            raise RuntimeError('缺少 blending 或 stacking 结果')
        yp_bl = np.asarray(r_bl.get('y_pred'), dtype=float)
        yp_st = np.asarray(r_st.get('y_pred'), dtype=float)
        yt_any = np.asarray(r_bl.get('y_true', y_true_used), dtype=float)
        idx_bl = r_bl.get('indices', None)
        kdf_bl = keys_df.iloc[idx_bl] if (idx_bl is not None and isinstance(keys_df, pd.DataFrame)) else keys_df
        if not isinstance(kdf_bl, pd.DataFrame):
            raise RuntimeError('缺少键数据用于选择代表性电池')
        # 电池ID选择
        if 'barcode' in kdf_bl.columns:
            ids = kdf_bl['barcode'].astype(str).values
        else:
            ids = (np.asarray(groups)[idx_bl] if (idx_bl is not None and len(groups) >= len(yt_any)) else np.array(['group'] * len(yt_any))).astype(str)
        # 计算每ID误差（MAE）
        df_eval = pd.DataFrame({'id': ids, 'pred': yp_bl, 'true': yt_any})
        g = df_eval.groupby('id')
        mae = g.apply(lambda d: float(np.mean(np.abs(d['true'].values - d['pred'].values))))
        ids_sorted = mae.sort_values().index.tolist()
        rep_id = ids_sorted[len(ids_sorted)//2] if ids_sorted else ids[0]
        mask = (ids == rep_id)
        cycles_rep = kdf_bl['cycle'].values[mask] if 'cycle' in kdf_bl.columns else np.arange(int(mask.sum()))
        yt_rep = yt_any[mask]
        yp_bl_rep = yp_bl[mask]
        # 对齐 stacking 的子集（索引可能不同）
        idx_st = r_st.get('indices', None)
        kdf_st = keys_df.iloc[idx_st] if (idx_st is not None and isinstance(keys_df, pd.DataFrame)) else keys_df
        if 'barcode' in kdf_st.columns:
            ids_st = kdf_st['barcode'].astype(str).values
        else:
            ids_st = (np.asarray(groups)[idx_st] if (idx_st is not None and len(groups) >= len(yt_any)) else np.array(['group'] * len(yt_any))).astype(str)
        mask_st = (ids_st == rep_id)
        yp_st_rep = yp_st[mask_st]
        cycles_st_rep = kdf_st['cycle'].values[mask_st] if 'cycle' in kdf_st.columns else np.arange(int(mask_st.sum()))
        # 为避免错配，按循环号合并两条预测与真实
        # 获取容量保持率真值；若不可用则跳过
        cap_true_rep = None
        if str(target_type).lower() == 'capacity_retention':
            cap_true_rep = yt_rep
        elif isinstance(X, pd.DataFrame) and 'capacity_retention_pct' in X.columns:
            try:
                cap_true_rep = np.asarray((X['capacity_retention_pct'].values)[:len(yt_rep)], dtype=float)
            except Exception:
                cap_true_rep = None
        if cap_true_rep is None:
            raise RuntimeError('Skip representative compare: capacity_true unavailable')
        cap_true_rep = np.clip(np.asarray(cap_true_rep, dtype=float), 0.0, 100.0)
        yp_bl_rep = np.clip(np.asarray(yp_bl_rep, dtype=float), 0.0, 100.0)
        df_st = pd.DataFrame({'cycle': cycles_st_rep, 'stacking': yp_st_rep})
        df_merge = pd.merge(df_bl, df_st, on='cycle', how='inner').sort_values('cycle')
        # 替换 true 列为裁剪后的容量保持率
        df_merge['true'] = cap_true_rep[:len(df_merge)]
        df_merge['blending'] = np.clip(np.asarray(df_merge['blending'], dtype=float), 0.0, 100.0)
        df_merge['stacking'] = np.clip(np.asarray(df_merge['stacking'], dtype=float), 0.0, 100.0)
        # 导出CSV与图
        out_base = os.path.join(art_dir, 'capacity_curve_compare_blending_stacking')
        df_merge.to_csv(out_base + '.csv', index=False, encoding='utf-8-sig')
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(df_merge['cycle'], df_merge['true'], color='#4C72B0', linewidth=1.6, label=_label_text('True', '真实值', label_lang))
        ax.plot(df_merge['cycle'], df_merge['blending'], color='#55A868', linewidth=1.6, linestyle='--', label='Blending')
        ax.plot(df_merge['cycle'], df_merge['stacking'], color='#C44E52', linewidth=1.6, linestyle='-.', label='Stacking')
        ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
        ax.set_ylabel(_label_text('Discharge Capacity (Ah)', '放电容量(Ah)', label_lang))
        ax.set_ylim(0, 100)
        try:
            ax.axhline(80.0, color='gray', linestyle=':', linewidth=1)
        except Exception:
            pass
        ax.legend(); ax.grid(True, alpha=0.3)
        _save_plot(fig, out_base)
    except Exception:
        pass

    # 保存选择特征列表（使用 selector 的重要度）
    imp_arr = np.asarray(getattr(selector, 'aggregated_importance_', np.zeros(len(getattr(selector, 'feature_names_', [])))))
    feat_names = list(getattr(selector, 'feature_names_', []))
    imp_map = {n: float(imp_arr[i]) for i, n in enumerate(feat_names)} if len(imp_arr) == len(feat_names) else {}
    sel_rows = [{'feature': f, 'importance': imp_map.get(f, 0.0)} for f in getattr(selector, 'selected_features_', [])]
    pd.DataFrame(sel_rows).sort_values('importance', ascending=False).to_csv(os.path.join(art_dir, 'selected_features.csv'), index=False, encoding='utf-8-sig')
    
    # RUL预测：使用SOH模型预测未来容量 + 计算RUL
    if str(target_type).lower() == 'rul':
        try:
            print("开始RUL预测：使用SOH模型预测未来容量衰減曲线")
            # 1. 重建 SOH数据集用于预测未来容量
            # 跳过：列表中的build_dataset_from_excel既然存在了，再次导入会候實用
            if data_mode == 'dir' and os.path.isdir(xlsx_dir):
                X_soh, y_soh, groups_soh, keys_soh = build_dataset_from_directory(xlsx_dir, target_type='capacity_retention', capacity_threshold_pct=threshold_pct, return_keys=True)
            else:
                X_soh, y_soh, keys_soh = build_dataset_from_excel(xlsx_path, target_type='capacity_retention', capacity_threshold_pct=threshold_pct, return_keys=True)
                groups_soh = None
                
            # 2. 使用训练好的最佳模型预测未来容量
            from sklearn.base import clone as _clone
            best_pipeline = _clone(best_result.get('pipeline'))
            if best_pipeline is not None:
                # **使用完整的X_full和y_full重新训练**
                best_pipeline.fit(X_full, y_full)  # 重新训练最佳模型
                    
                # 3. 预测每个电池未来的容量保持率
                if isinstance(keys_soh, pd.DataFrame) and 'barcode' in keys_soh.columns and 'cycle' in keys_soh.columns:
                    rul_results = []
                    for barcode in pd.unique(keys_soh['barcode']):
                        mask = keys_soh['barcode'] == barcode
                        if mask.sum() < 10:  # 至少需要10个周期的数据
                            continue
                            
                        # 获取该电池的数据
                        X_batt = X_soh[mask]
                        keys_batt = keys_soh[mask].sort_values('cycle')
                        y_true_batt = y_soh[mask] if len(y_soh) >= mask.sum() else None
                            
                        # 用SOH模型预测未来N个周期的容量
                        try:
                            capacity_pred = best_pipeline.predict(X_batt)
                        except Exception as e:
                            print(f"电池 {barcode} 容量预测失败: {e}")
                            continue
                            
                        # 构建预测结果DataFrame
                        df_pred = pd.DataFrame({
                            'cycle': keys_batt['cycle'].values,
                            'barcode': barcode,
                            'predicted_capacity_pct': np.clip(capacity_pred, 0, 120),
                            'true_capacity_pct': np.clip(y_true_batt, 0, 120) if y_true_batt is not None else np.nan
                        }).sort_values('cycle')
                            
                        # 4. 外推找到80%失效点
                        thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
                        eol_pos = np.where(df_pred['predicted_capacity_pct'] < thr_pct)[0]
                        if len(eol_pos) > 0:
                            eol_cycle = float(df_pred['cycle'].iloc[eol_pos[0]])
                        else:
                            # 如果预测容量从未低于阈值，则使用最大周期
                            eol_cycle = float(df_pred['cycle'].max())
                            
                        # 计算每个周期的RUL
                        df_pred['predicted_rul'] = np.maximum(0.0, eol_cycle - df_pred['cycle'])
                        
                        # 保存结果
                        rul_results.append(df_pred)
                        
                        # 保存单个电池的预测结果
                        battery_rul_dir = os.path.join(art_dir, 'by_battery')
                        os.makedirs(battery_rul_dir, exist_ok=True)
                        df_pred.to_csv(os.path.join(battery_rul_dir, f'{barcode}_rul_prediction.csv'), index=False, encoding='utf-8-sig')
                        
                    # 保存总体RUL预测结果
                    if rul_results:
                        all_rul_df = pd.concat(rul_results, ignore_index=True)
                        all_rul_df.to_csv(os.path.join(art_dir, 'rul_prediction_detailed.csv'), index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"RUL预测过程出错: {e}")
            import traceback
            traceback.print_exc()

    # 拟合并保存最优管道（如可用）
    try:
        from sklearn.base import clone as _clone
        
        # 保存best_model
        best_pipeline = _clone(best_result.get('pipeline'))
        if best_pipeline is not None:
            # **使用完整的X_full和y_full重新训练**
            best_pipeline.fit(X_full, y_full)
            joblib.dump(best_pipeline, os.path.join(art_dir, 'best_model.pkl'))
            print(f"✓ 已保存best_model.pkl: {best_name}")
        
        # 保存stacking和blending模型
        for result in results:
            model_name = result.get('name', '')
            if model_name in ['stacking', 'blending']:
                pipeline = _clone(result.get('pipeline'))
                if pipeline is not None:
                    # 使用完整数据重新训练
                    pipeline.fit(X_full, y_full)
                    model_path = os.path.join(art_dir, f'{model_name}_model.pkl')
                    joblib.dump(pipeline, model_path)
                    print(f"✓ 已保存{model_name}_model.pkl")
    except Exception as e:
        print(f"保存模型失败: {e}")

    # 保存配置
    cfg = {
        'xlsx_path': (xlsx_path if data_mode == 'file' else None),
        'xlsx_dir': (xlsx_dir if data_mode == 'dir' else None),
        'target_type': target_type,
        'capacity_threshold_pct': threshold_pct,
        'cv_strategy': cv_strategy,
        'cv_splits': cv_splits,
        'best_model': best_name,
        'random_state': 42,
        'selected_features_count': int(len(getattr(selector, 'selected_features_', []))),
        'n_groups': (int(len(pd.unique(groups))) if groups is not None else None),
    }
    save_config(cfg, art_dir)

    # 图像
    save_corr_heatmap(corr_pred, 'Feature vs Prediction Correlation', os.path.join(art_dir, 'feature_vs_pred_corr_heatmap.png'))
    save_corr_heatmap(corr_true, 'Feature vs True Target Correlation', os.path.join(art_dir, 'feature_vs_true_corr_heatmap.png'))
    save_literature_style_corr_matrix(X_sel_full, y, 'Feature Correlation Matrix', os.path.join(art_dir, 'literature_style_corr_matrix.png'))
    try:
        _y_true_for_plot = y_true_used if str(target_type).lower() == 'capacity_retention' else None
        _ = plot_capacity_compare_ah(xlsx_path, data_mode, y_pred_oof, art_dir, label_lang, keys_df=keys_df, y_true_used=_y_true_for_plot, pred_label=best_name)
    except Exception:
        pass

# 从现有 artifacts CSV 重绘 RUL 图（无需重新训练）
def remake_rul_plots_from_csv(base_dir):
    import os
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    def _label_text(en, zh, lang):
        return zh if str(lang).lower().startswith('zh') else en
    label_lang = os.environ.get('LABEL_LANG', 'zh')
    art_dir = os.path.join(base_dir, 'rul')
    byb = os.path.join(art_dir, 'by_battery')
    # 总体: RUL预测改为显示容量保持率曲线
    p_all = os.path.join(art_dir, 'rul_curve.csv')
    if os.path.isfile(p_all):
        try:
            df = pd.read_csv(p_all)
            df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values('cycle')
            # RUL预测：显示容量保持率曲线
            fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            ax.plot(df['cycle'], df['rul_true'], color='#4C72B0', linewidth=1.6, label=_label_text('True Capacity', '真实容量%'))
            ax.plot(df['cycle'], df['rul_pred'], color='#55A868', linewidth=1.6, linestyle='--', label=_label_text('Predicted Capacity', '预测容量%'))
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('Capacity Retention (%)', '容量保持率(%)', label_lang))
            y_top = float(max(100.0, 1.05 * max(np.nanmax(df['rul_true']), np.nanmax(df['rul_pred']))))
            ax.set_ylim(0, y_top)
            try:
                thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
                ax.axhline(thr_pct, color='gray', linestyle=':', linewidth=1, label=_label_text('EOL Threshold', '失效阈值'))
            except Exception:
                pass
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(art_dir, 'capacity_curve_from_rul.png'), dpi=180)
        except Exception as e:
            print(f"重绘总体容量曲线失败: {e}")
    # 每电池: RUL预测改为显示容量保持率曲线
    for p in glob.glob(os.path.join(byb, '*_rul_curve.csv')):
        try:
            df = pd.read_csv(p)
            df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values('cycle')
            # RUL预测：显示容量保持率曲线
            fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
            ax.plot(df['cycle'], df['rul_true'], color='#4C72B0', linewidth=1.6, label=_label_text('True Capacity', '真实容量%'))
            ax.plot(df['cycle'], df['rul_pred'], color='#55A868', linewidth=1.6, linestyle='--', label=_label_text('Predicted Capacity', '预测容量%'))
            ax.set_xlabel(_label_text('Cycle', '循环次数', label_lang))
            ax.set_ylabel(_label_text('Capacity Retention (%)', '容量保持率(%)', label_lang))
            y_top = float(max(100.0, 1.05 * max(np.nanmax(df['rul_true']), np.nanmax(df['rul_pred']))))
            ax.set_ylim(0, y_top)
            try:
                thr_pct = float(os.environ.get('CAPACITY_THRESHOLD_PCT', '80.0'))
                ax.axhline(thr_pct, color='gray', linestyle=':', linewidth=1, label=_label_text('EOL Threshold', '失效阈值'))
            except Exception:
                pass
            ax.legend(); ax.grid(True, alpha=0.3)
            base = os.path.splitext(p)[0]
            fig.savefig(base + '_capacity_curve.png', dpi=180)
        except Exception as e:
            print(f"重绘电池容量曲线失败 {p}: {e}")

if __name__ == '__main__':
    import os
    try:
        if str(os.environ.get('REMAKE_RUL_PLOTS_FROM_CSV','0')).strip() == '1':
            project_root = os.path.dirname(os.path.abspath(__file__))
            base = os.path.join(os.path.dirname(project_root), 'artifacts')
            remake_rul_plots_from_csv(base)
    except Exception:
        pass
