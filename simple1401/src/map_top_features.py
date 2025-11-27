import os
import re
import pandas as pd
import joblib

# 解析 artifacts 目录的辅助函数
def resolve_artifacts_dir():
    import os
    env_dir = os.environ.get('ARTIFACTS_DIR', '').strip()
    candidates = []
    if env_dir:
        candidates.append(env_dir)
    cwd = os.getcwd()
    candidates += [
        os.path.join(cwd, 'artifacts'),
        os.path.join(os.path.dirname(cwd), 'artifacts'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts')),
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            print(f'使用ARTIFACTS目录: {d}')
            return d
    return env_dir or os.path.join(os.path.dirname(cwd), 'artifacts')

def load_selected_feature_names(art_dir: str):
    import os, joblib
    pkl_path = os.path.join(art_dir, 'best_model.pkl')
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f'未找到 {pkl_path}')
    pipe = joblib.load(pkl_path)
    sel = pipe.named_steps.get('select', None)
    if sel is None:
        raise RuntimeError('管道中未找到 select 步骤')
    names = list(getattr(sel, 'selected_features_', []))
    if not names:
        raise RuntimeError('selected_features_ 为空，无法映射 f_i -> 原始特征名')
    return names

def attach_original_names(corr_df, ordered_feature_names):
    import pandas as pd
    out = corr_df.copy()
    try:
        idx = out['feature'].astype(str).str.extract(r'^f_(\d+)$')[0].astype(int)
        out['original_name'] = [
            ordered_feature_names[i] if i < len(ordered_feature_names) else None
            for i in idx
        ]
    except Exception:
        out['original_name'] = None
    return out

def map_and_save(art_dir: str, csv_name: str, out_name: str, top_k: int = 20):
    import os, pandas as pd, joblib
    src = os.path.join(art_dir, csv_name)
    if not os.path.isfile(src):
        print(f'未找到 {src}，跳过')
        return None
    ordered_feature_names = None
    try:
        ordered_feature_names = load_selected_feature_names(art_dir)
    except Exception as e:
        print(f'读取selected_features失败: {e}')
    df = pd.read_csv(src)
    if 'correlation' in df.columns:
        df = df.reindex(df['correlation'].abs().sort_values(ascending=False).index)
    df_top = df.head(top_k)
    if ordered_feature_names:
        df_top = attach_original_names(df_top, ordered_feature_names)
    out_path = os.path.join(art_dir, out_name)
    df_top.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已保存 {csv_name} 的 Top-{top_k} 映射到: {out_path}')
    print(df_top)
    return df_top

# === 新增：从数据集直接计算原始特征名的相关性 ===
def load_config(art_dir: str):
    import os, json, re
    cfg_yaml = os.path.join(art_dir, 'config.yaml')
    cfg_json = os.path.join(art_dir, 'config.json')
    if os.path.isfile(cfg_yaml):
        try:
            import yaml
            with open(cfg_yaml, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            # 简易 YAML 解析器（无第三方库时兜底）
            cfg = {}
            with open(cfg_yaml, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or ':' not in line:
                        continue
                    k, v = line.split(':', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # 试图解析成数值
                    if re.fullmatch(r'-?\d+', v):
                        v = int(v)
                    elif re.fullmatch(r'-?\d+\.\d+', v):
                        v = float(v)
                    elif v.lower() in ('true','false'):
                        v = (v.lower() == 'true')
                    cfg[k] = v
            return cfg
    if os.path.isfile(cfg_json):
        with open(cfg_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise FileNotFoundError('未找到 config.yaml/config.json')

def load_dataset_by_config(cfg):
    from build_dataset import build_dataset_from_excel
    from train import build_dataset_from_directory
    xlsx_dir = cfg.get('xlsx_dir') or ''
    xlsx_path = cfg.get('xlsx_path') or ''
    target_type = cfg.get('target_type', 'RUL')
    threshold_pct = float(cfg.get('capacity_threshold_pct', 80.0))
    if xlsx_dir and os.path.isdir(xlsx_dir):
        X, y, groups = build_dataset_from_directory(xlsx_dir, target_type=target_type, capacity_threshold_pct=threshold_pct)
    elif xlsx_path and os.path.isfile(xlsx_path):
        X, y = build_dataset_from_excel(xlsx_path, target_type=target_type, capacity_threshold_pct=threshold_pct)
        groups = None
    else:
        raise FileNotFoundError('配置中未找到有效的数据路径')
    return X, y, groups

def compute_from_dataset(art_dir: str, top_k: int = 20):
    import os, joblib, pandas as pd
    from sklearn.model_selection import cross_val_predict
    from train import compute_feature_corr_df
    print('尝试使用原始数据集直接计算“特征名×相关性”…')
    cfg = load_config(art_dir)
    X, y, groups = load_dataset_by_config(cfg)
    pkl_path = os.path.join(art_dir, 'best_model.pkl')
    pipe = joblib.load(pkl_path)
    # 用OOF预测更稳健；无法分组时退化到普通CV
    try:
        y_pred = cross_val_predict(pipe, X, y, cv=cfg.get('cv_splits', 5), n_jobs=1, method='predict')
    except Exception:
        y_pred = pipe.fit(X, y).predict(X)
    corr_pred = compute_feature_corr_df(X, y_pred, method='spearman')
    corr_true = compute_feature_corr_df(X, y, method='spearman')
    # 取Top-K并保存
    corr_pred_top = corr_pred.reindex(corr_pred['correlation'].abs().sort_values(ascending=False).index).head(top_k)
    corr_true_top = corr_true.reindex(corr_true['correlation'].abs().sort_values(ascending=False).index).head(top_k)
    out_pred = os.path.join(art_dir, 'top20_feature_pred_names.csv')
    out_true = os.path.join(art_dir, 'top20_feature_true_names.csv')
    corr_pred_top.to_csv(out_pred, index=False, encoding='utf-8-sig')
    corr_true_top.to_csv(out_true, index=False, encoding='utf-8-sig')
    print('原始特征名 Top-20（预测相关性）：')
    print(corr_pred_top[['feature', 'correlation', 'p_value']])
    print('原始特征名 Top-20（真值相关性）：')
    print(corr_true_top[['feature', 'correlation', 'p_value']])

# === 新增：利用管道 + 数据集做“局部f_k -> 原始列名”的精准映射 ===
def map_corr_with_pipeline(art_dir: str, csv_name: str, out_name: str, top_k: int = 20):
    import os, re, joblib, numpy as np
    from typing import List
    # 1) 加载管道与数据集
    pkl_path = os.path.join(art_dir, 'best_model.pkl')
    if not os.path.isfile(pkl_path):
        print(f'未找到 {pkl_path}，跳过管道映射')
        return None
    pipe = joblib.load(pkl_path)
    cfg = load_config(art_dir)
    X, y, groups = load_dataset_by_config(cfg)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 2) 构建 var 之后的原始列名列表（post_var_names）
    imputer = pipe.named_steps.get('imputer', None)
    var = pipe.named_steps.get('var', None)
    try:
        X_imp = imputer.transform(X) if imputer is not None else X.values
    except Exception:
        # 兜底：用每列中位数填充
        X_imp = X.copy()
        med = X_imp.median(numeric_only=True)
        for c in X_imp.columns:
            if c in med.index:
                X_imp[c] = X_imp[c].fillna(med[c])
            else:
                X_imp[c] = X_imp[c].fillna(0.0)
        X_imp = X_imp.fillna(0.0).values
    # 支持索引的获取（优先用已拟合的 variances_）
    if var is not None and hasattr(var, 'variances_'):
        thr = float(getattr(var, 'threshold', 0.0))
        support_idx = [i for i, v in enumerate(var.variances_) if float(v) > thr]
    else:
        thr = float(getattr(var, 'threshold', 0.0)) if var is not None else 0.0
        var_arr = np.var(X_imp, axis=0)
        support_idx = [i for i, v in enumerate(var_arr) if float(v) > thr]
    post_var_names: List[str] = [X.columns[i] for i in support_idx]

    # 新增：应用 corr 过滤后的列名，保证与 select 的索引对齐
    corr_step = pipe.named_steps.get('corr', None)
    if corr_step is not None and hasattr(corr_step, 'keep_indices_') and corr_step.keep_indices_ is not None:
        post_corr_names = [post_var_names[i] for i in corr_step.keep_indices_]
    else:
        post_corr_names = post_var_names
    # 3) 取选择器的“绝对编号”列表，位置顺序与 transform 输出一致
    sel = pipe.named_steps.get('select', None)
    if sel is None or not hasattr(sel, 'selected_features_') or not sel.selected_features_:
        print('管道中 select 步骤不可用或未选择到特征，跳过映射')
        return None
    abs_idx_list = []
    for name in sel.selected_features_:
        m = re.match(r'^f_(\d+)$', str(name))
        abs_idx_list.append(int(m.group(1)) if m else None)

    # 4) 读取相关性 CSV（列名是“局部编号”f_k），Top-K 排序
    src = os.path.join(art_dir, csv_name)
    if not os.path.isfile(src):
        print(f'未找到 {src}，跳过')
        return None
    df = pd.read_csv(src)
    if 'correlation' in df.columns:
        df = df.reindex(df['correlation'].abs().sort_values(ascending=False).index)
    df_top = df.head(top_k).copy()

    # 5) 做局部 f_k -> 绝对编号 -> 原始列名 的映射
    def _map_original_name(f_str):
        m = re.match(r'^f_(\d+)$', str(f_str))
        if not m:
            return str(f_str)
        local_idx = int(m.group(1))
        if local_idx < len(abs_idx_list):
            abs_idx = abs_idx_list[local_idx]
            if abs_idx is not None and abs_idx < len(post_corr_names):
                return post_corr_names[abs_idx]
        return None

    df_top['original_name'] = df_top['feature'].apply(_map_original_name)
    out_path = os.path.join(art_dir, out_name)
    df_top.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'用管道+数据集推断的原始列名 Top-{top_k} 已保存: {out_path}')
    try:
        print(df_top[['feature','original_name','correlation','p_value']])
    except Exception:
        print(df_top)
    return df_top

if __name__ == '__main__':
    art_dir = resolve_artifacts_dir()
    print(f'ARTIFACTS_DIR = {art_dir}')
    pred_df = map_and_save(art_dir, 'feature_vs_pred_corr.csv', 'top20_feature_pred_names.csv', top_k=20)
    true_df = map_and_save(art_dir, 'feature_vs_true_corr.csv', 'top20_feature_true_names.csv', top_k=20)
    # 若原方法映射为 None 或伪名，改用“管道+数据集”的精准映射
    def _needs_fallback(df):
        try:
            if df is None:
                return True
            if 'original_name' not in df.columns:
                return True
            s = df['original_name'].astype(str)
            # 改成“只要出现伪名/缺失就触发”
            if s.isna().any():
                return True
            if s.str.match(r'^f_\d+$').any():
                return True
            return False
        except Exception:
            return True

    if _needs_fallback(pred_df):
        pred_df = map_corr_with_pipeline(art_dir, 'feature_vs_pred_corr.csv', 'top20_feature_pred_names.csv', top_k=20)
    if _needs_fallback(true_df):
        true_df = map_corr_with_pipeline(art_dir, 'feature_vs_true_corr.csv', 'top20_feature_true_names.csv', top_k=20)

    # 最后兜底：如果仍旧失败，直接用原始数据集计算真实列名版
    def _still_bad(df):
        try:
            return (df is None) or ('original_name' in df.columns and (df['original_name'].isna().all() or df['original_name'].astype(str).str.match(r'^f_\d+$').all()))
        except Exception:
            return True

    if _still_bad(pred_df) or _still_bad(true_df):
        compute_from_dataset(art_dir, top_k=20)
