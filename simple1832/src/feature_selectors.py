import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

try:
    from skrebate import RReliefF
except Exception:
    RReliefF = None

try:
    import shap
except Exception:
    shap = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


# 顶部（import 后）新增一个安全替换函数
def safe_nan_to_num(arr, nan_val=0.0, pos_val=1e6, neg_val=-1e6):
    arr = np.asarray(arr)
    mask_nan = np.isnan(arr)
    if mask_nan.any():
        arr = arr.copy()
        arr[mask_nan] = nan_val
    mask_inf = np.isinf(arr)
    if mask_inf.any():
        signs = np.sign(arr[mask_inf])
        arr[mask_inf] = np.where(signs >= 0, pos_val, neg_val)
    return arr

class CombinedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_features: Optional[int] = None,
        relief_neighbors: int = 100,
        weights: Tuple[float, float, float] = (0.33, 0.33, 0.34),
        correlation_threshold: float = 0.9,
        random_state: int = 42,
        xgb_params: Optional[dict] = None,
        use_permutation_if_shap_fails: bool = True,
    ):
        self.n_features = n_features
        self.relief_neighbors = relief_neighbors
        self.weights = weights
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        self.use_permutation_if_shap_fails = use_permutation_if_shap_fails
        self.xgb_params = xgb_params or {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'n_jobs': -1,
            'random_state': random_state
        }

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            _arr = np.asarray(X)
            if _arr.ndim == 1:
                _arr = _arr.reshape(-1, 1)
            X_df = pd.DataFrame(_arr, columns=[f'f_{i}' for i in range(_arr.shape[1])])
        self.feature_names_ = list(X_df.columns)
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_np = pd.to_numeric(np.asarray(y), errors='coerce')

        finite_y_mask = np.isfinite(y_np)
        X_df = X_df.loc[finite_y_mask]
        y_np = y_np[finite_y_mask]

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_np = imputer.fit_transform(X_df)
        self.imputer_ = imputer
        X_np = safe_nan_to_num(X_np, nan_val=0.0, pos_val=1e6, neg_val=-1e6)

        if len(X_np) < 10:
            self.selected_features_ = self.feature_names_.copy()
            self.aggregated_importance_ = np.ones(len(self.feature_names_))
            return self

        try:
            # ReliefF
            if RReliefF is not None:
                rel = RReliefF(
                    n_features=X_np.shape[1],
                    n_neighbors=min(self.relief_neighbors, max(5, X_np.shape[0] // 10))
                )
                rel.fit(X_np, y_np)
                imp_rel = safe_nan_to_num(np.asarray(rel.feature_importances_), nan_val=0.0)
            else:
                imp_rel = np.zeros(X_np.shape[1], dtype=float)

            # SHAP via XGB 或回退置换重要性
            if XGBRegressor is not None and shap is not None:
                xgb = XGBRegressor(**self.xgb_params)
                xgb.fit(X_np, y_np)
                explainer = shap.TreeExplainer(xgb)
                shap_vals = explainer.shap_values(X_np)
                imp_shap = safe_nan_to_num(np.abs(shap_vals).mean(axis=0), nan_val=0.0)
            else:
                perm = permutation_importance(
                    LinearRegression().fit(X_np, y_np),
                    X_np, y_np,
                    scoring='neg_mean_squared_error',
                    n_repeats=5, random_state=self.random_state
                )
                imp_shap = safe_nan_to_num(np.abs(perm.importances_mean), nan_val=0.0)

            # LassoCV 稀疏系数
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_np)
            lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=20000)
            lasso.fit(X_std, y_np)
            imp_lasso = safe_nan_to_num(np.abs(lasso.coef_), nan_val=0.0)

            # 相关性兜底
            corr = safe_nan_to_num(
                np.abs([np.corrcoef(X_np[:, i], y_np)[0, 1] for i in range(X_np.shape[1])]),
                nan_val=0.0
            )

            # 归一与加权融合
            def _norm(v):
                v = np.asarray(v, dtype=float)
                s = np.linalg.norm(v) + 1e-12
                return v / s

            w_rel, w_shap, w_lasso = self.weights
            agg = w_rel * _norm(imp_rel) + w_shap * _norm(imp_shap) + w_lasso * _norm(imp_lasso)
            self.aggregated_importance_ = 0.5 * _norm(agg) + 0.5 * _norm(corr)

            # 保存每路重要性
            self.imp_rel_ = imp_rel
            self.imp_shap_ = imp_shap
            self.imp_lasso_ = imp_lasso
            self.corr_importance_ = corr

            # 选 top-k
            k = (self.n_features or X_np.shape[1])
            top_idx = np.argsort(self.aggregated_importance_)[-k:]
            self.selected_features_ = [self.feature_names_[i] for i in top_idx]
            self.selected_indices_ = list(top_idx)
            return self
        except Exception:
            # CombinedFeatureSelector.fit(...) 内相关性计算片段
            def _compute_corr_safe(x_col, y_vec):
                if np.std(x_col) < 1e-10:
                    return 0.0
                c = np.corrcoef(x_col, y_vec)[0, 1]
                return 0.0 if np.isnan(c) else float(c)
            
            corr = np.array([
                _compute_corr_safe(X_np[:, i], y_np) for i in range(X_np.shape[1])
            ], dtype=float)

            self.aggregated_importance_ = corr
            if self.n_features and self.n_features < len(self.feature_names_):
                top_indices = np.argsort(corr)[-self.n_features:]
                self.selected_features_ = [self.feature_names_[i] for i in top_indices]
                self.selected_indices_ = list(top_indices)
            else:
                self.selected_features_ = self.feature_names_.copy()
                self.selected_indices_ = list(range(len(self.feature_names_)))
            return self

    def transform(self, X):
        """转换数据，只保留选择的特征"""
        try:
            if not hasattr(self, 'selected_features_'):
                raise ValueError("必须先调用fit方法")

            if isinstance(X, pd.DataFrame):
                X_df = X.copy()
                X_df = X_df.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
                X_df = X_df.reindex(columns=self.feature_names_, fill_value=np.nan)
                if hasattr(self, 'imputer_'):
                    X_imp = self.imputer_.transform(X_df)
                    arr = np.asarray(X_imp)
                else:
                    arr = X_df.fillna(0.0).values
            else:
                arr = np.asarray(X)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arr = np.where(np.isfinite(arr), arr, np.nan)
                if hasattr(self, 'imputer_') and arr.shape[1] == len(self.feature_names_):
                    X_df = pd.DataFrame(arr, columns=self.feature_names_)
                    X_imp = self.imputer_.transform(X_df)
                    arr = np.asarray(X_imp)
                else:
                    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

            idx = getattr(self, 'selected_indices_', None)
            if idx is None:
                idx = [self.feature_names_.index(f) for f in self.selected_features_ if f in self.feature_names_]
            idx = [i for i in idx if i < arr.shape[1]]
            if not idx:
                return arr
            return arr[:, idx]
        except Exception:
            return X if isinstance(X, np.ndarray) else np.asarray(X)


# CorrelationFilter：按列相关性阈值去冗余
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95, method: str = 'pearson'):
        self.threshold = threshold
        self.method = method

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            self.feature_names_in_ = list(df.columns)
        else:
            arr = X if isinstance(X, np.ndarray) else np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            df = pd.DataFrame(arr, columns=[f'f_{i}' for i in range(arr.shape[1])])
            self.feature_names_in_ = list(df.columns)

        # 计算列间相关性矩阵，缺失置零以稳健
        corr_df = df.corr(method=('spearman' if str(self.method).lower() == 'spearman' else 'pearson'))
        corr = np.nan_to_num(corr_df.values, nan=0.0)

        n = corr.shape[0]
        to_drop = set()
        keep = []
        for i in range(n):
            if i in to_drop:
                continue
            keep.append(i)
            for j in range(i + 1, n):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) >= self.threshold:
                    to_drop.add(j)

        self.keep_indices_ = keep
        self.n_features_in_ = n
        return self

    def transform(self, X):
        if not hasattr(self, 'keep_indices_') or self.keep_indices_ is None:
            return X
        arr = X.values if hasattr(X, 'values') else np.asarray(X)
        arr_out = arr[:, self.keep_indices_]
        if isinstance(X, pd.DataFrame):
            cols = [self.feature_names_in_[i] for i in self.keep_indices_]
            return pd.DataFrame(arr_out, columns=cols)
        return arr_out

    def get_support(self, indices: bool = False):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        for i in (self.keep_indices_ or []):
            mask[i] = True
        return (self.keep_indices_ if indices else mask)

    def get_params(self, deep=True):
        return {'threshold': self.threshold, 'method': self.method}
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
