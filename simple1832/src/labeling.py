import pandas as pd
import numpy as np

def _find_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low_cols = {str(c).lower(): c for c in cols}
    for c in candidates:
        cl = str(c).lower()
        for k, orig in low_cols.items():
            if cl in k:
                return orig
    return None

def compute_rul_from_sheet1(s1: pd.DataFrame, threshold_pct: float = 80.0) -> pd.DataFrame:
    """
    基于容量保持率计算RUL，确保：
    1. EOL点为真实80%容量保持率点（而非数据集末尾）
    2. 使用Savitzky-Golay平滑避免噪声干扰
    3. 支持线性插值获取精确EOL周期
    """
    df = s1.copy()
    cycle_col_name = _find_col(df, ['cycle','循环号','循环编号','Cycle','Cycle No'])
    if cycle_col_name:
        df['循环号'] = pd.to_numeric(df[cycle_col_name], errors='coerce')
        df.dropna(subset=['循环号'], inplace=True)
        df['循环号'] = df['循环号'].astype(int)
    else:
        df['循环号'] = np.arange(1, len(df) + 1)
    cap_col_name = _find_col(df, ['capacity_retention_pct','容量保持率(%)','容量保持率','SOH(%)','SOH','Capacity Retention'])
    if cap_col_name is None:
        raise ValueError(f'未找到容量保持率列。可用列: {list(df.columns)}')
    df['容量保持率'] = pd.to_numeric(df[cap_col_name], errors='coerce')
    df.dropna(subset=['容量保持率'], inplace=True)
    df = df.sort_values('循环号').reset_index(drop=True)
    
    # 异常值检测和处理（使用移动中位数过滤离群点）
    retention_vals = df['容量保持率'].values
    retention_vals_cleaned = retention_vals.copy()
    rolling_median = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).median()
    rolling_std = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).std()
    outlier_mask = np.abs(retention_vals - rolling_median) > (3 * rolling_std)
    outlier_mask = outlier_mask.fillna(False).values
    if outlier_mask.sum() > 0:
        retention_vals_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
    
    # 平滑容量保持率曲线（Savitzky-Golay滤波）
    import os
    from scipy.signal import savgol_filter
    n_points = len(retention_vals_cleaned)
    if n_points >= 11:
        window = min(11, n_points if n_points % 2 == 1 else n_points - 1)
        poly_order = 2
        smoothed = savgol_filter(retention_vals_cleaned, window_length=window, polyorder=poly_order, mode='nearest')
        # 二次平滑
        smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
    elif n_points >= 5:
        window = min(5, n_points if n_points % 2 == 1 else n_points - 1)
        poly_order = 2
        smoothed = savgol_filter(retention_vals_cleaned, window_length=window, polyorder=poly_order, mode='nearest')
    else:
        smoothed = retention_vals_cleaned
    
    # 用平滑后的值更新容量保持率（这样后续计算RUL时使用的是平滑后的值）
    df['容量保持率'] = smoothed
    
    # 查找真实EOL点（平滑后首次持续低于阈值）
    consecutive = int(os.environ.get('EOL_CONSECUTIVE_POINTS', '3'))
    m = (smoothed <= threshold_pct).astype(int)
    run = pd.Series(m).rolling(window=max(1, consecutive), min_periods=1).sum().values
    pos = np.where(run >= max(1, consecutive))[0]
    
    if pos.size > 0 and pos[0] > 0:
        # 线性插值获取精确EOL周期
        i = int(pos[0])
        val_before = float(smoothed[i-1])
        val_after = float(smoothed[i])
        cyc_before = float(df['循环号'].iloc[i-1])
        cyc_after = float(df['循环号'].iloc[i])
        
        if val_before > threshold_pct >= val_after and val_before != val_after:
            # 线性插值
            t = (threshold_pct - val_before) / (val_after - val_before)
            eol_cycle = cyc_before + t * (cyc_after - cyc_before)
        else:
            eol_cycle = float(cyc_after)
    elif pos.size > 0:
        eol_cycle = float(df['循环号'].iloc[pos[0]])
    else:
        # 若全程未达到阈值，使用最后一个周期
        eol_cycle = float(df['循环号'].max())
    
    # 计算所有cycle的RUL（EOL后的RUL为0）
    df['RUL'] = np.maximum(0.0, eol_cycle - df['循环号'])
    
    out = df[['循环号','RUL']].rename(columns={'循环号':'cycle'})
    barcode_col_name = _find_col(df, ['barcode','条码','Barcode','Cell ID'])
    if barcode_col_name:
        out['barcode'] = df[barcode_col_name].astype(str)
        return out[['cycle','barcode','RUL']]
    return out[['cycle','RUL']]

def compute_rul_from_first_discharge_capacity(s1: pd.DataFrame, threshold_pct: float = 80.0) -> pd.DataFrame:
    """
    基于首周期放电容量计算RUL，逐电池处理：
    1. 计算容量保持率 = 当前容量 / 首周期容量 * 100
    2. 平滑容量保持率曲线
    3. 精确定位真实EOL点（80%阈值）
    """
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter
    
    df = s1.copy()
    cycle_col_name = _find_col(df, ['cycle','循环号','循环编号','Cycle','Cycle No'])
    if cycle_col_name:
        df['循环号'] = pd.to_numeric(df[cycle_col_name], errors='coerce')
        df.dropna(subset=['循环号'], inplace=True)
        df['循环号'] = df['循环号'].astype(int)
    else:
        df['循环号'] = np.arange(1, len(df) + 1)
    barcode_col_name = _find_col(df, ['barcode','条码','Barcode','Cell ID'])
    if barcode_col_name:
        df['条码'] = df[barcode_col_name].astype(str)
    cap_col_name = _find_col(df, ['discharge_capacity_ah','放电容量(Ah)','放电容量','Discharge Capacity','Capacity(Ah)'])
    if cap_col_name is None:
        raise ValueError(f'未找到放电容量列。可用列: {list(df.columns)}')
    cap_vals = pd.to_numeric(df[cap_col_name], errors='coerce')
    df = df.assign(_cap=cap_vals)
    if '条码' in df.columns:
        df = df.sort_values(['条码', '循环号'])
        first_cap = df.groupby('条码')['_cap'].transform(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan)
    else:
        df = df.sort_values('循环号')
        first_valid = df['_cap'].dropna()
        first_cap = pd.Series(float(first_valid.iloc[0]) if not first_valid.empty else np.nan, index=df.index)
    df['retention_pct'] = 100.0 * df['_cap'] / first_cap
    df = df.dropna(subset=['retention_pct']).copy()
    df = df[first_cap > 0].copy()
    out = []
    import os
    consecutive = int(os.environ.get('EOL_CONSECUTIVE_POINTS', '3'))
    
    if '条码' in df.columns:
        for bc, g in df.groupby('条码', sort=False):
            g = g.sort_values('循环号').reset_index(drop=True)
            retention_vals = g['retention_pct'].values
            n_points = len(retention_vals)
            
            # 异常值检测和处理
            retention_vals_cleaned = retention_vals.copy()
            rolling_median = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).median()
            rolling_std = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).std()
            outlier_mask = np.abs(retention_vals - rolling_median) > (3 * rolling_std)
            outlier_mask = outlier_mask.fillna(False).values
            if outlier_mask.sum() > 0:
                retention_vals_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
            
            # 平滑容量保持率曲线
            if n_points >= 11:
                window = min(11, n_points if n_points % 2 == 1 else n_points - 1)
                smoothed = savgol_filter(retention_vals_cleaned, window_length=window, polyorder=2, mode='nearest')
                smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
            elif n_points >= 5:
                window = min(5, n_points if n_points % 2 == 1 else n_points - 1)
                smoothed = savgol_filter(retention_vals_cleaned, window_length=window, polyorder=2, mode='nearest')
            else:
                smoothed = retention_vals_cleaned
            
            # 用平滑后的值更新retention_pct（注意：需要更新df中对应的行）
            g = g.copy()
            g['retention_pct'] = smoothed
            
            # 精确定位EOL
            m = (smoothed <= threshold_pct).astype(int)
            run = pd.Series(m).rolling(window=max(1, consecutive), min_periods=1).sum().values
            pos = np.where(run >= max(1, consecutive))[0]
            
            if pos.size > 0 and pos[0] > 0:
                i = int(pos[0])
                val_before = float(smoothed[i-1])
                val_after = float(smoothed[i])
                cyc_before = float(g['循环号'].iloc[i-1])
                cyc_after = float(g['循环号'].iloc[i])
                
                if val_before > threshold_pct >= val_after and val_before != val_after:
                    t = (threshold_pct - val_before) / (val_after - val_before)
                    eol_cycle = cyc_before + t * (cyc_after - cyc_before)
                else:
                    eol_cycle = float(cyc_after)
            elif pos.size > 0:
                eol_cycle = float(g['循环号'].iloc[pos[0]])
            else:
                eol_cycle = float(g['循环号'].max())
            
            # 计算所有cycle的RUL（EOL后的RUL为0）
            g['RUL'] = np.maximum(0.0, eol_cycle - g['循环号'])
            out.append(pd.DataFrame({'cycle': g['循环号'], 'barcode': bc, 'RUL': g['RUL']}))
        return pd.concat(out, ignore_index=True)
    else:
        g = df.sort_values('循环号').reset_index(drop=True)
        retention_vals = g['retention_pct'].values
        n_points = len(retention_vals)
        
        # 异常值检测和处理
        retention_vals_cleaned = retention_vals.copy()
        rolling_median = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).median()
        rolling_std = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).std()
        outlier_mask = np.abs(retention_vals - rolling_median) > (3 * rolling_std)
        outlier_mask = outlier_mask.fillna(False).values
        if outlier_mask.sum() > 0:
            retention_vals_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
        
        if n_points >= 11:
            window = min(11, n_points if n_points % 2 == 1 else n_points - 1)
            smoothed = savgol_filter(retention_vals_cleaned, window_length=window, polyorder=2, mode='nearest')
            smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
        elif n_points >= 5:
            window = min(5, n_points if n_points % 2 == 1 else n_points - 1)
            smoothed = savgol_filter(retention_vals_cleaned, window_length=window, polyorder=2, mode='nearest')
        else:
            smoothed = retention_vals_cleaned
        
        # 用平滑后的值更新retention_pct
        g = g.copy()
        g['retention_pct'] = smoothed
        
        m = (smoothed <= threshold_pct).astype(int)
        run = pd.Series(m).rolling(window=max(1, consecutive), min_periods=1).sum().values
        pos = np.where(run >= max(1, consecutive))[0]
        
        if pos.size > 0 and pos[0] > 0:
            i = int(pos[0])
            val_before = float(smoothed[i-1])
            val_after = float(smoothed[i])
            cyc_before = float(g['循环号'].iloc[i-1])
            cyc_after = float(g['循环号'].iloc[i])
            
            if val_before > threshold_pct >= val_after and val_before != val_after:
                t = (threshold_pct - val_before) / (val_after - val_before)
                eol_cycle = cyc_before + t * (cyc_after - cyc_before)
            else:
                eol_cycle = float(cyc_after)
        elif pos.size > 0:
            eol_cycle = float(g['循环号'].iloc[pos[0]])
        else:
            eol_cycle = float(g['循环号'].max())
        
        # 计算所有cycle的RUL（EOL后的RUL为0）
        g['RUL'] = np.maximum(0.0, eol_cycle - g['循环号'])
        return pd.DataFrame({'cycle': g['循环号'], 'RUL': g['RUL']})

def compute_rul_with_smoothing(s1: pd.DataFrame, threshold_pct: float = 80.0, window: int = 9, poly: int = 2, consecutive: int = 3) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    df = s1.copy()
    cyc_col = _find_col(df, ['cycle','循环号','循环编号','Cycle','Cycle No'])
    if cyc_col:
        df['cycle_num'] = pd.to_numeric(df[cyc_col], errors='coerce')
        df.dropna(subset=['cycle_num'], inplace=True)
        df['cycle_num'] = df['cycle_num'].astype(float)
    else:
        df['cycle_num'] = np.arange(1, len(df) + 1, dtype=float)
    bc_col = _find_col(df, ['barcode','条码','Barcode','Cell ID'])
    if bc_col:
        df['bc'] = df[bc_col].astype(str)
    cap_col = _find_col(df, ['discharge_capacity_ah','放电容量(Ah)','放电容量','Discharge Capacity','Capacity(Ah)'])
    ret_col = _find_col(df, ['capacity_retention_pct','容量保持率(%)','容量保持率','SOH(%)','SOH','Capacity Retention'])
    if ret_col is None and cap_col is None:
        raise ValueError('容量保持率或放电容量缺失')
    if ret_col is None:
        df['_cap'] = pd.to_numeric(df[cap_col], errors='coerce')
        if 'bc' in df.columns:
            df = df.sort_values(['bc','cycle_num'])
            base = df.groupby('bc')['_cap'].transform(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan)
        else:
            df = df.sort_values('cycle_num')
            s0 = df['_cap'].dropna()
            base = pd.Series(float(s0.iloc[0]) if not s0.empty else np.nan, index=df.index)
        df['ret_pct'] = 100.0 * pd.to_numeric(df['_cap'], errors='coerce') / base
    else:
        df['ret_pct'] = pd.to_numeric(df[ret_col], errors='coerce')
        df = df.sort_values('cycle_num')
    x = df['ret_pct'].values
    
    # 步骤1：异常值检测和处理（使用移动中位数过滤离群点）
    x_cleaned = x.copy()
    rolling_median = pd.Series(x).rolling(window=7, min_periods=3, center=True).median()
    rolling_std = pd.Series(x).rolling(window=7, min_periods=3, center=True).std()
    
    # 识别离群点（超过3倍标准差的点）
    outlier_mask = np.abs(x - rolling_median) > (3 * rolling_std)
    outlier_mask = outlier_mask.fillna(False).values
    
    # 用移动中位数替换离群点
    if outlier_mask.sum() > 0:
        x_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
    
    # 步骤2：强平滑（Savitzky-Golay滤波）
    try:
        from scipy.signal import savgol_filter
        # 增大窗口以获得更强平滑效果
        win = int(max(11, window))  # 从5提高到11
        if win % 2 == 0:
            win += 1
        # 对清理后的数据进行平滑
        x_s = savgol_filter(np.nan_to_num(x_cleaned, nan=np.nanmedian(x_cleaned)), win, int(max(1, poly)))
    except Exception:
        # 备用方案：使用移动中位数平滑
        x_s = pd.Series(x_cleaned).rolling(window=max(7, consecutive*2+1), min_periods=3, center=True).median().fillna(method='bfill').fillna(method='ffill').values
    
    # 步骤3：二次平滑（移动平均进一步平滑）
    x_s = pd.Series(x_s).rolling(window=5, min_periods=1, center=True).mean().values
    
    # 用平滑后的值更新ret_pct
    df['ret_pct'] = x_s
    
    m = (x_s <= float(threshold_pct)).astype(int)
    run = pd.Series(m).rolling(window=max(1, consecutive), min_periods=1).sum().values
    pos = np.where(run >= max(1, consecutive))[0]
    if pos.size > 0 and pos[0] > 0:
        i = int(pos[0])
        x0, x1 = float(x_s[i-1]), float(x_s[i])
        c0, c1 = float(df['cycle_num'].iloc[i-1]), float(df['cycle_num'].iloc[i])
        if np.isfinite(x0) and np.isfinite(x1) and x1 != x0:
            t = (threshold_pct - x0) / (x1 - x0)
            eol_cycle = float(c0 + t * (c1 - c0))
        else:
            eol_cycle = float(df['cycle_num'].iloc[i])
    else:
        eol_cycle = float(df['cycle_num'].max())
    # 计算所有cycle的RUL（EOL后的RUL为0）
    df['RUL'] = np.clip(eol_cycle - df['cycle_num'], 0.0, None)
    out = df[['cycle_num','RUL']].rename(columns={'cycle_num':'cycle'})
    if 'bc' in df.columns:
        out['barcode'] = df['bc']
        return out[['cycle','barcode','RUL']]
    return out[['cycle','RUL']]
