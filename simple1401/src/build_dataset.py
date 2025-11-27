import pandas as pd
import numpy as np
from excel_loader import read_battery_excel
from feature_engineering import features_sheet1, features_sheet2, features_sheet3, merge_feature_tables
from labeling import compute_rul_from_sheet1

def read_battery_csv(path):
    """
    读取CSV格式的电池数据文件（自动编码探测，回退常见编码）
    """
    if not path.endswith('.csv'):
        raise ValueError('文件必须是CSV格式')
    enc = None
    try:
        from charset_normalizer import from_path as cn_from_path
        res = cn_from_path(path).best()
        if res and res.encoding:
            enc = res.encoding
    except Exception:
        enc = None
    if enc is None:
        try:
            import chardet
            with open(path, 'rb') as f:
                raw = f.read(65536)
            enc = chardet.detect(raw).get('encoding')
        except Exception:
            enc = None
    tried = []
    for ec in [enc, 'utf-8', 'gb18030', 'gbk', 'latin-1']:
        if not ec:
            continue
        try:
            df = pd.read_csv(path, encoding=ec)
            break
        except Exception:
            tried.append(ec)
            df = None
    if df is None:
        raise UnicodeDecodeError('read_csv', b'', 0, 1, f'编码探测失败，已尝试: {tried}')
    # 清理列名
    df.columns = [str(c).strip() for c in df.columns]
    return df.copy(), df.copy(), df.copy()

def build_dataset_from_excel(path: str, target_type: str = 'RUL', capacity_threshold_pct: float = 80.0, return_keys: bool = False):
    # 检查文件类型
    if path.endswith('.csv'):
        s1, s2, s3 = read_battery_csv(path)
    else:
        s1, s2, s3 = read_battery_excel(path)
    
    # **关键修改**: 在提取特征之前，先对原始容量数据进行平滑处理
    # 这样提取的特征就是基于平滑后的数据，与平滑后的目标一致
    import os
    if os.environ.get('SMOOTH_RAW_CAPACITY', '1').lower() in ('1', 'true', 'yes', 'on'):
        # 查找容量列
        cap_col = None
        for col_candidate in ['discharge_capacity_ah', 'capacity_ah', '放电容量(Ah)', 'Capacity(Ah)']:
            if col_candidate in s1.columns:
                cap_col = col_candidate
                break
        
        if cap_col is not None:
            # 对容量列进行平滑处理（按barcode分组，如果有的话）
            s1_smoothed = s1.copy()
            s1_smoothed[cap_col] = pd.to_numeric(s1_smoothed[cap_col], errors='coerce')
            
            if 'barcode' in s1_smoothed.columns or '条码' in s1_smoothed.columns:
                barcode_col = 'barcode' if 'barcode' in s1_smoothed.columns else '条码'
                s1_smoothed = s1_smoothed.sort_values([barcode_col, 'cycle'] if 'cycle' in s1_smoothed.columns else [barcode_col]).copy()
                smoothed_capacity = []
                
                for bc, g in s1_smoothed.groupby(barcode_col, sort=False):
                    cap_vals = g[cap_col].values
                    # 异常值检测和处理
                    cap_cleaned = cap_vals.copy()
                    if len(cap_vals) >= 7:
                        rolling_median = pd.Series(cap_vals).rolling(window=7, min_periods=3, center=True).median()
                        rolling_std = pd.Series(cap_vals).rolling(window=7, min_periods=3, center=True).std()
                        outlier_mask = np.abs(cap_vals - rolling_median) > (3 * rolling_std)
                        outlier_mask = outlier_mask.fillna(False).values
                        if outlier_mask.sum() > 0:
                            cap_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
                    # 平滑处理
                    if len(cap_cleaned) >= 11:
                        try:
                            from scipy.signal import savgol_filter
                            window = min(11, len(cap_cleaned) if len(cap_cleaned) % 2 == 1 else len(cap_cleaned) - 1)
                            smoothed = savgol_filter(cap_cleaned, window_length=window, polyorder=2, mode='nearest')
                            # 二次平滑
                            smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
                        except Exception:
                            smoothed = cap_cleaned
                    elif len(cap_cleaned) >= 5:
                        smoothed = pd.Series(cap_cleaned).rolling(window=5, min_periods=1, center=True).mean().values
                    else:
                        smoothed = cap_cleaned
                    smoothed_capacity.extend(smoothed)
                
                s1_smoothed[cap_col] = smoothed_capacity
            else:
                # 单电池，直接平滑整体容量序列
                cap_vals = s1_smoothed[cap_col].values
                cap_cleaned = cap_vals.copy()
                if len(cap_vals) >= 7:
                    rolling_median = pd.Series(cap_vals).rolling(window=7, min_periods=3, center=True).median()
                    rolling_std = pd.Series(cap_vals).rolling(window=7, min_periods=3, center=True).std()
                    outlier_mask = np.abs(cap_vals - rolling_median) > (3 * rolling_std)
                    outlier_mask = outlier_mask.fillna(False).values
                    if outlier_mask.sum() > 0:
                        cap_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
                if len(cap_cleaned) >= 11:
                    try:
                        from scipy.signal import savgol_filter
                        window = min(11, len(cap_cleaned) if len(cap_cleaned) % 2 == 1 else len(cap_cleaned) - 1)
                        smoothed = savgol_filter(cap_cleaned, window_length=window, polyorder=2, mode='nearest')
                        smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
                    except Exception:
                        smoothed = cap_cleaned
                elif len(cap_cleaned) >= 5:
                    smoothed = pd.Series(cap_cleaned).rolling(window=5, min_periods=1, center=True).mean().values
                else:
                    smoothed = cap_cleaned
                s1_smoothed[cap_col] = smoothed
            
            # 用平滑后的s1替换原始s1，后续特征提取都基于平滑数据
            s1 = s1_smoothed
    
    # 现在基于平滑后的s1提取特征
    f1 = features_sheet1(s1)
    f2 = features_sheet2(s2)
    f3 = features_sheet3(s3)
    X = merge_feature_tables(f1, f2, f3)
    # 新增：并入 CEEMDAN 分解特征（来自第一页容量序列）
    from feature_engineering import features_ceemdan_capacity
    f_emd = features_ceemdan_capacity(s1)
    keys = [k for k in ['cycle','barcode'] if k in X.columns and k in f_emd.columns]
    if keys:
        X = pd.merge(X, f_emd, on=keys, how='left')
    if target_type.upper() == 'RUL':
        import os
        from labeling import compute_rul_from_sheet1, compute_rul_from_first_discharge_capacity, compute_rul_with_smoothing
        rul_base = os.environ.get('RUL_BASE', 'first_discharge_capacity').lower()
        use_smooth = str(os.environ.get('RUL_SMOOTH', '1')).strip().lower() in ('1','true','on','yes')
        if use_smooth:
            ytab = compute_rul_with_smoothing(s1, threshold_pct=capacity_threshold_pct)
        else:
            if rul_base in ('first_discharge_capacity','capacity_first','discharge_capacity'):
                ytab = compute_rul_from_first_discharge_capacity(s1, threshold_pct=capacity_threshold_pct)
            else:
                ytab = compute_rul_from_sheet1(s1, threshold_pct=capacity_threshold_pct)
        keys = [k for k in ['cycle','barcode'] if k in X.columns and k in ytab.columns]
        X = pd.merge(X, ytab, on=keys, how='inner')
        # 清理RUL：非负与有限
        X['RUL'] = pd.to_numeric(X['RUL'], errors='coerce')
        m = X['RUL'].replace([np.inf, -np.inf], np.nan).notna() & (X['RUL'] >= 0.0)
        X = X[m]
        # 去除重复键（保留第一条RUL真值）
        X = X.drop_duplicates(subset=keys, keep='first')
        y = X['RUL'].values
        X = X.drop(columns=['RUL'])
    elif target_type == 'capacity_retention':
        if 'capacity_retention_pct' not in X.columns:
            dc_col = None
            for cand in ['discharge_capacity_ah']:
                if cand in X.columns:
                    dc_col = cand
                    break
            if dc_col is None:
                raise ValueError('X中缺少capacity_retention_pct，且无法找到放电容量列用于计算')
            s = pd.to_numeric(X[dc_col], errors='coerce')
            if 'barcode' in X.columns:
                try:
                    base = X.groupby('barcode')[dc_col].transform(lambda v: pd.to_numeric(v, errors='coerce').dropna().iloc[0] if pd.to_numeric(v, errors='coerce').dropna().shape[0] else np.nan)
                except Exception:
                    base = s.iloc[0]
            else:
                base = s.dropna().iloc[0] if s.dropna().shape[0] else np.nan
            X['capacity_retention_pct'] = (s / base) * 100.0
            X['capacity_retention_pct'] = X['capacity_retention_pct'].replace([np.inf, -np.inf], np.nan)
        X['capacity_retention_pct'] = pd.to_numeric(X['capacity_retention_pct'], errors='coerce')
        X['capacity_retention_pct'] = X['capacity_retention_pct'].replace([np.inf, -np.inf], np.nan)
        import os
        soh_min = float(os.environ.get('SOH_MIN', '0'))
        soh_max = float(os.environ.get('SOH_MAX', '110'))
        m = X['capacity_retention_pct'].between(soh_min, soh_max)
        X = X[m]
        
        # 对SOH任务也进行异常值处理和平滑（按电池分组）
        if 'barcode' in X.columns and os.environ.get('SOH_SMOOTH_OUTLIERS', '1').lower() in ('1', 'true', 'yes', 'on'):
            X = X.sort_values(['barcode', 'cycle']).copy()
            # 先保存原始值用于对比
            X['capacity_retention_pct_original'] = X['capacity_retention_pct'].values
            smoothed_retention = []
            for bc, g in X.groupby('barcode', sort=False):
                retention_vals = g['capacity_retention_pct'].values
                # 异常值检测和处理
                retention_cleaned = retention_vals.copy()
                if len(retention_vals) >= 7:
                    rolling_median = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).median()
                    rolling_std = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).std()
                    outlier_mask = np.abs(retention_vals - rolling_median) > (3 * rolling_std)
                    outlier_mask = outlier_mask.fillna(False).values
                    if outlier_mask.sum() > 0:
                        retention_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
                # 平滑处理
                if len(retention_cleaned) >= 11:
                    try:
                        from scipy.signal import savgol_filter
                        window = min(11, len(retention_cleaned) if len(retention_cleaned) % 2 == 1 else len(retention_cleaned) - 1)
                        smoothed = savgol_filter(retention_cleaned, window_length=window, polyorder=2, mode='nearest')
                        # 二次平滑
                        smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
                    except Exception:
                        smoothed = retention_cleaned
                elif len(retention_cleaned) >= 5:
                    smoothed = pd.Series(retention_cleaned).rolling(window=5, min_periods=1, center=True).mean().values
                else:
                    smoothed = retention_cleaned
                smoothed_retention.extend(smoothed)
            # 用平滑后的值替换原始值（这样keys_df中保存的就是平滑后的值）
            X['capacity_retention_pct'] = smoothed_retention
            y = X['capacity_retention_pct'].values
        else:
            y = X['capacity_retention_pct'].values
    else:
        raise ValueError('不支持的target_type')
    non_feature_keys = [c for c in ['cycle','barcode'] if c in X.columns]
    # RUL任务也要保留capacity_retention_pct，用于后续计算真实RUL
    if target_type.upper() == 'RUL' and 'capacity_retention_pct' in X.columns:
        non_feature_keys.append('capacity_retention_pct')
    # SOH任务也保留capacity_retention_pct，用于绘制真实曲线
    if target_type == 'capacity_retention' and 'capacity_retention_pct' in X.columns:
        non_feature_keys.append('capacity_retention_pct')
    keys_df = X[non_feature_keys].copy() if non_feature_keys else pd.DataFrame(index=X.index)
    X = X.drop(columns=non_feature_keys)
    X = X.select_dtypes(include=['number'])
    if return_keys:
        return X, y, keys_df
    return X, y
