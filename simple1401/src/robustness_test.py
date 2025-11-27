import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def test_noise_robustness(model, X, y, noise_levels=[0.01, 0.05, 0.10]):
    """测试对噪声的鲁棒性，返回RMSE随噪声变化的表"""
    X_np = X.values if hasattr(X, 'values') else np.asarray(X)
    results = []
    for level in noise_levels:
        X_noisy = X_np + np.random.normal(0, level * X_np.std(axis=0), X_np.shape)
        y_pred = model.predict(X_noisy)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        results.append({'noise_level': float(level), 'rmse': rmse})
    return pd.DataFrame(results)

def test_extrapolation(model, X, y, cycle_col_index=None):
    """测试时间外推能力：仅用前80%训练，后20%测试，返回R2"""
    X_np = X.values if hasattr(X, 'values') else np.asarray(X)
    y_np = np.asarray(y)
    split_idx = int(len(X_np) * 0.8)
    model.fit(X_np[:split_idx], y_np[:split_idx])
    y_pred = model.predict(X_np[split_idx:])
    return float(r2_score(y_np[split_idx:], y_pred))