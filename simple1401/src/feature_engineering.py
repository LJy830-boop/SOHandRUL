import numpy as np
import pandas as pd

def _to_seconds(s):
    try:
        return pd.to_timedelta(s.astype(str), errors='coerce').dt.total_seconds()
    except Exception:
        return pd.Series(np.nan, index=s.index if hasattr(s, 'index') else None)

def _num(s):
    return pd.to_numeric(s, errors='coerce')

def _find_col(df: pd.DataFrame, candidates, fallback_contains=True):
    cols = list(df.columns)
    # 直接匹配
    for c in candidates:
        if c in cols:
            return c
    if fallback_contains:
        # 不区分大小写的包含匹配
        low_cols = {c.lower(): c for c in cols}
        for c in candidates:
            cl = c.lower()
            for k, orig in low_cols.items():
                if cl in k:
                    return orig
    return None

CYCLE_COL_CANDIDATES = ['cycle','Cycle','Cycle No','Cycle_No','Cycle_Number','CycleIndex','Cycle_Index']
BARCODE_COL_CANDIDATES = ['barcode','Barcode','Cell ID','Battery ID']
def _clean_and_prepare_df(df_in: pd.DataFrame):
    df = df_in.copy()
    cycle_col = _find_col(df, CYCLE_COL_CANDIDATES)
    if cycle_col:
        df['cycle'] = pd.to_numeric(df[cycle_col], errors='coerce')
        df.dropna(subset=['cycle'], inplace=True)
        df['cycle'] = df['cycle'].astype(int)
    else:
        df['cycle'] = np.arange(1, len(df) + 1)
    barcode_col = _find_col(df, BARCODE_COL_CANDIDATES)
    if barcode_col:
        df['barcode'] = df[barcode_col].astype(str)
    return df

def features_sheet1(s1: pd.DataFrame) -> pd.DataFrame:
    df = _clean_and_prepare_df(s1)
    keys = ['cycle']
    if 'barcode' in df.columns:
        keys.append('barcode')
    out = df[keys].drop_duplicates().copy()
    for col in [
        'charge_capacity_ah','discharge_capacity_ah','charge_energy_Wh','discharge_energy_Wh','charge_avg_voltage_V','discharge_avg_voltage_V',
        'charge_begin_voltage_V','charge_end_voltage_V','discharge_begin_voltage_V','discharge_end_voltage_V','coulombic_efficiency_pct','capacity_retention_pct',
        'energy_efficiency_pct','dcir_mohm','plateau_capacity1_ah','rest_voltage_after_discharge','cc_charge_ratio_pct','cc_charge_capacity_ah',
        'charge_median_voltage_V','discharge_median_voltage_V','net_discharge_capacity_ah','net_discharge_energy_Wh'
    ]:
        if col in df.columns:
            df[col] = _num(df[col])
    for tcol in ['plateau_time1','charge_time','discharge_time']:
        if tcol in df.columns:
            df[tcol+'_s'] = _to_seconds(df[tcol])
    charge_Q = 'charge_capacity_ah' if 'charge_capacity_ah' in df.columns else None
    discharge_Q = 'discharge_capacity_ah' if 'discharge_capacity_ah' in df.columns else None
    charge_E = 'charge_energy_Wh' if 'charge_energy_Wh' in df.columns else None
    discharge_E = 'discharge_energy_Wh' if 'discharge_energy_Wh' in df.columns else None
    if charge_Q and discharge_Q:
        df['coulombic_eff'] = (df[discharge_Q] / df[charge_Q]).replace([np.inf, -np.inf], np.nan)
    if charge_E and discharge_E:
        df['energy_efficiency'] = (df[discharge_E] / df[charge_E]).replace([np.inf, -np.inf], np.nan)
    def _rng(a, b):
        if a in df.columns and b in df.columns:
            return (_num(df[b]) - _num(df[a]))
        return pd.Series(np.nan, index=df.index)
    if 'discharge_begin_voltage_V' in df.columns or 'discharge_end_voltage_V' in df.columns:
        df['discharge_voltage_range'] = _rng('discharge_begin_voltage_V', 'discharge_end_voltage_V')
    if 'charge_begin_voltage_V' in df.columns or 'charge_end_voltage_V' in df.columns:
        df['charge_voltage_range'] = _rng('charge_begin_voltage_V', 'charge_end_voltage_V')
    if 'charge_time' in df.columns:
        df['charge_time'] = _to_seconds(df['charge_time'])
    if 'discharge_time' in df.columns:
        df['discharge_time'] = _to_seconds(df['discharge_time'])
    if 'charge_time' in df.columns and 'discharge_time' in df.columns:
        df['charge_discharge_time_ratio'] = (df['charge_time'] / df['discharge_time']).replace([np.inf, -np.inf], np.nan)
    if {'charge_avg_voltage_V','discharge_avg_voltage_V'} <= set(df.columns):
        out['avg_voltage_hysteresis'] = df['charge_avg_voltage_V'] - df['discharge_avg_voltage_V']
    if {'charge_begin_voltage_V','charge_end_voltage_V'} <= set(df.columns):
        out['charge_voltage_delta'] = df['charge_end_voltage_V'] - df['charge_begin_voltage_V']
    if {'discharge_begin_voltage_V','discharge_end_voltage_V'} <= set(df.columns):
        out['discharge_voltage_delta'] = df['discharge_begin_voltage_V'] - df['discharge_end_voltage_V']
    if 'coulombic_efficiency_pct' in df.columns:
        out['coulombic_efficiency_pct'] = df['coulombic_efficiency_pct']
    if 'energy_efficiency_pct' in df.columns:
        out['energy_efficiency_pct'] = df['energy_efficiency_pct']
    if 'capacity_retention_pct' in df.columns:
        df['capacity_retention_pct'] = _num(df['capacity_retention_pct']).replace([np.inf, -np.inf], np.nan)
        import os
        soh_min = float(os.environ.get('SOH_MIN', '0'))
        soh_max = float(os.environ.get('SOH_MAX', '110'))
        m_valid = (df['capacity_retention_pct'].between(soh_min, soh_max))
        df.loc[~m_valid, 'capacity_retention_pct'] = np.nan
        out['capacity_retention_pct'] = df.groupby(keys)['capacity_retention_pct'].first().reset_index(drop=True)
    elif discharge_Q:
        if 'barcode' in df.columns:
            df_srt = df.sort_values(['barcode','cycle'])
            base = df_srt.groupby('barcode')[discharge_Q].transform(lambda s: pd.to_numeric(s, errors='coerce').dropna().iloc[0] if pd.to_numeric(s, errors='coerce').dropna().shape[0] else np.nan)
            dc_num = _num(df[discharge_Q])
            df['capacity_retention_pct'] = 100.0 * dc_num / base
        else:
            df_srt = df.sort_values('cycle')
            first_valid = _num(df_srt[discharge_Q]).dropna()
            base_val = float(first_valid.iloc[0]) if not first_valid.empty else np.nan
            dc_num = _num(df[discharge_Q])
            df['capacity_retention_pct'] = 100.0 * dc_num / base_val
        df['capacity_retention_pct'] = df['capacity_retention_pct'].replace([np.inf, -np.inf], np.nan)
        import os
        soh_min = float(os.environ.get('SOH_MIN', '0'))
        soh_max = float(os.environ.get('SOH_MAX', '110'))
        m_valid = df['capacity_retention_pct'].between(soh_min, soh_max)
        df.loc[~m_valid, 'capacity_retention_pct'] = np.nan
        out['capacity_retention_pct'] = df.groupby(keys)['capacity_retention_pct'].first().reset_index(drop=True)
    # RUL相关派生：到80%阈值的距离与滚动趋势
    if 'capacity_retention_pct' in df.columns:
        if 'barcode' in df.columns:
            df_srt = df.sort_values(['barcode','cycle'])
            df['soh_gap_to_80_pct'] = df_srt['capacity_retention_pct'] - 80.0
            try:
                roll_slope = df_srt.groupby('barcode')['capacity_retention_pct'].transform(
                    lambda s: s.rolling(5, min_periods=3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1), raw=False)
                )
            except Exception:
                roll_slope = pd.Series(np.nan, index=df.index)
            df['soh_roll_slope_pct_w5'] = roll_slope
            df['soh_roll_accel_pct_w5'] = df_srt.groupby('barcode')['soh_roll_slope_pct_w5'].transform(lambda s: s - s.shift(1))
            df['soh_roll_var_pct_w5'] = df_srt.groupby('barcode')['capacity_retention_pct'].transform(lambda s: s.rolling(5, min_periods=3).var())

            # 新增：W=20的长期趋势特征
            try:
                roll_slope_20 = df_srt.groupby('barcode')['capacity_retention_pct'].transform(
                    lambda s: s.rolling(20, min_periods=10).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1), raw=False)
                )
            except Exception:
                roll_slope_20 = pd.Series(np.nan, index=df.index)
            df['soh_roll_slope_pct_w20'] = roll_slope_20
            df['soh_roll_accel_pct_w20'] = df_srt.groupby('barcode')['soh_roll_slope_pct_w20'].transform(lambda s: s - s.shift(1))
            df['soh_roll_var_pct_w20'] = df_srt.groupby('barcode')['capacity_retention_pct'].transform(lambda s: s.rolling(20, min_periods=10).var())
            
            if 'discharge_avg_voltage_V' in df.columns:
                 df['volt_roll_var_w20'] = df_srt.groupby('barcode')['discharge_avg_voltage_V'].transform(lambda s: s.rolling(20, min_periods=10).var())
        else:
            df_srt = df.sort_values('cycle')
            df['soh_gap_to_80_pct'] = df_srt['capacity_retention_pct'] - 80.0
            try:
                roll_slope = df_srt['capacity_retention_pct'].rolling(5, min_periods=3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1), raw=False)
            except Exception:
                roll_slope = pd.Series(np.nan, index=df.index)
            df['soh_roll_slope_pct_w5'] = roll_slope.values
            df['soh_roll_accel_pct_w5'] = pd.Series(df['soh_roll_slope_pct_w5']).diff().values
            df['soh_roll_var_pct_w5'] = df_srt['capacity_retention_pct'].rolling(5, min_periods=3).var().values

            # 新增：W=20的长期趋势特征
            try:
                roll_slope_20 = df_srt['capacity_retention_pct'].rolling(20, min_periods=10).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / max(len(x) - 1, 1), raw=False)
            except Exception:
                roll_slope_20 = pd.Series(np.nan, index=df.index)
            df['soh_roll_slope_pct_w20'] = roll_slope_20.values
            df['soh_roll_accel_pct_w20'] = pd.Series(df['soh_roll_slope_pct_w20']).diff().values
            df['soh_roll_var_pct_w20'] = df_srt['capacity_retention_pct'].rolling(20, min_periods=10).var().values

            if 'discharge_avg_voltage_V' in df.columns:
                 df['volt_roll_var_w20'] = df_srt['discharge_avg_voltage_V'].rolling(20, min_periods=10).var().values

        out['soh_gap_to_80_pct'] = df.groupby(keys)['soh_gap_to_80_pct'].first().reset_index(drop=True)
        out['soh_roll_slope_pct_w5'] = df.groupby(keys)['soh_roll_slope_pct_w5'].first().reset_index(drop=True)
        out['soh_roll_accel_pct_w5'] = df.groupby(keys)['soh_roll_accel_pct_w5'].first().reset_index(drop=True)
        out['soh_roll_var_pct_w5'] = df.groupby(keys)['soh_roll_var_pct_w5'].first().reset_index(drop=True)
        
        # 注册新特征到输出
        out['soh_roll_slope_pct_w20'] = df.groupby(keys)['soh_roll_slope_pct_w20'].first().reset_index(drop=True)
        out['soh_roll_accel_pct_w20'] = df.groupby(keys)['soh_roll_accel_pct_w20'].first().reset_index(drop=True)
        out['soh_roll_var_pct_w20'] = df.groupby(keys)['soh_roll_var_pct_w20'].first().reset_index(drop=True)
        if 'volt_roll_var_w20' in df.columns:
            out['volt_roll_var_w20'] = df.groupby(keys)['volt_roll_var_w20'].first().reset_index(drop=True)
        if discharge_Q:
            if 'barcode' in df.columns:
                df_srt = df.sort_values(['barcode','cycle'])
                base80 = df_srt.groupby('barcode')[discharge_Q].transform(lambda s: pd.to_numeric(s, errors='coerce').dropna().iloc[0] * 0.8 if pd.to_numeric(s, errors='coerce').dropna().shape[0] else np.nan)
            else:
                sdis = pd.to_numeric(df[discharge_Q], errors='coerce').dropna()
                base80 = pd.Series(sdis.iloc[0] * 0.8 if sdis.shape[0] else np.nan, index=df.index)
            df['soh_gap_to_80_ah'] = pd.to_numeric(df[discharge_Q], errors='coerce') - base80
            out['soh_gap_to_80_ah'] = df.groupby(keys)['soh_gap_to_80_ah'].first().reset_index(drop=True)
    if 'dcir_mohm' in df.columns:
        out['dcir_mohm'] = df['dcir_mohm']
    if 'plateau_capacity1_ah' in df.columns:
        out['plateau_capacity1_ah'] = df['plateau_capacity1_ah']
    if 'rest_voltage_after_discharge' in df.columns:
        out['rest_voltage_after_discharge'] = df['rest_voltage_after_discharge']
    if 'cc_charge_ratio_pct' in df.columns:
        out['cc_charge_ratio_pct'] = df['cc_charge_ratio_pct']
    if 'cc_charge_capacity_ah' in df.columns:
        out['cc_charge_capacity_ah'] = df['cc_charge_capacity_ah']
    if 'charge_time_s' in df.columns:
        out['charge_time_s'] = df['charge_time_s']
    if 'discharge_time_s' in df.columns:
        out['discharge_time_s'] = df['discharge_time_s']
    if 'plateau_time1_s' in df.columns:
        out['plateau_time_s'] = df['plateau_time1_s']
    if {'net_discharge_energy_Wh','net_discharge_capacity_ah'} <= set(df.columns):
        out['net_energy_per_capacity'] = df['net_discharge_energy_Wh'] / (df['net_discharge_capacity_ah'] + 1e-12)
    for factor in ['coulombic_eff','energy_efficiency','discharge_voltage_range','charge_voltage_range','charge_discharge_time_ratio']:
        if factor in df.columns:
            out[factor] = df.groupby(keys)[factor].first().reset_index(drop=True)
    # 增加增量与归一化周期
    if 'dcir_mohm' in df.columns:
        if 'barcode' in df.columns:
            df['dcir_delta'] = df.sort_values(['barcode','cycle']).groupby('barcode')['dcir_mohm'].transform(lambda s: s.diff())
        else:
            df['dcir_delta'] = df.sort_values('cycle')['dcir_mohm'].diff()
        out['dcir_delta'] = df.groupby(keys)['dcir_delta'].first().reset_index(drop=True)
    if 'coulombic_efficiency_pct' in df.columns:
        if 'barcode' in df.columns:
            df['coulombic_efficiency_pct_delta'] = df.sort_values(['barcode','cycle']).groupby('barcode')['coulombic_efficiency_pct'].transform(lambda s: s.diff())
        else:
            df['coulombic_efficiency_pct_delta'] = df.sort_values('cycle')['coulombic_efficiency_pct'].diff()
        out['coulombic_efficiency_pct_delta'] = df.groupby(keys)['coulombic_efficiency_pct_delta'].first().reset_index(drop=True)
    if 'energy_efficiency_pct' in df.columns:
        if 'barcode' in df.columns:
            df['energy_efficiency_pct_delta'] = df.sort_values(['barcode','cycle']).groupby('barcode')['energy_efficiency_pct'].transform(lambda s: s.diff())
        else:
            df['energy_efficiency_pct_delta'] = df.sort_values('cycle')['energy_efficiency_pct'].diff()
        out['energy_efficiency_pct_delta'] = df.groupby(keys)['energy_efficiency_pct_delta'].first().reset_index(drop=True)
    if 'barcode' in df.columns:
        df['cycle_norm_in_battery'] = df.groupby('barcode')['cycle'].transform(lambda s: s / (s.max() if s.max() else 1))
    else:
        mx = df['cycle'].max() if 'cycle' in df.columns else None
        df['cycle_norm_in_battery'] = df['cycle'] / (mx if mx else 1)
    out['cycle_norm_in_battery'] = df.groupby(keys)['cycle_norm_in_battery'].first().reset_index(drop=True)
    for col in ['charge_capacity_ah','discharge_capacity_ah','charge_energy_Wh','discharge_energy_Wh','charge_median_voltage_V','discharge_median_voltage_V']:
        if col in df.columns:
            out[col] = df.groupby(keys)[col].first().reset_index(drop=True)
    return out.set_index(keys) if keys else out

def features_sheet2(s2: pd.DataFrame) -> pd.DataFrame:
    df = _clean_and_prepare_df(s2)
    keys = ['cycle']
    if 'barcode' in df.columns:
        keys.append('barcode')
    for col in [
        'capacity_ah','charge_capacity_ah','discharge_capacity_ah','energy_Wh','charge_energy_Wh','discharge_energy_Wh',
        'begin_voltage_V','end_voltage_V','median_voltage_V','dcir_mohm','begin_current_A','end_current_A',
        'charge_avg_voltage_V','discharge_avg_voltage_V','voltage_max_V','voltage_min_V','cc_charge_ratio_pct'
    ]:
        if col in df.columns:
            df[col] = _num(df[col])
    for tcol in ['step_time','charge_time','discharge_time']:
        if tcol in df.columns:
            df[tcol+'_s'] = _to_seconds(df[tcol])
    gb = df.groupby(keys, dropna=False)
    out = df[keys].drop_duplicates().copy()
    agg_candidates = {
        'charge_capacity_ah': 'sum',
        'discharge_capacity_ah': 'sum',
        'charge_energy_Wh': 'sum',
        'discharge_energy_Wh': 'sum',
        'dcir_mohm': ['mean','max','min'],
        'begin_current_A': 'mean',
        'end_current_A': 'mean',
        'voltage_max_V': 'max',
        'voltage_min_V': 'min',
        'cc_charge_ratio_pct': 'mean'
    }
    valid_agg = {k: v for k, v in agg_candidates.items() if k in df.columns}
    if valid_agg:
        agg_df = gb.agg(valid_agg)
        agg_df.columns = ['_'.join([c for c in col if c]) for col in agg_df.columns.to_flat_index()]
        agg_df = agg_df.reset_index()
        out = pd.merge(out, agg_df, on=keys, how='left')
    cols_time = ['step_time_s','charge_time_s','discharge_time_s']
    for name, src in [('step_time_charge_s','charge_time_s'),('step_time_discharge_s','discharge_time_s')]:
        if src in df.columns:
            s = df[[*keys, src]].groupby(keys)[src].sum()
            out = pd.merge(out, s.reset_index().rename(columns={src: name}), on=keys, how='left')
    if set(cols_time) <= set(out.columns):
        out['cycle_time_s'] = out['step_time_charge_s'].fillna(0) + out['step_time_discharge_s'].fillna(0)
    if {'voltage_max_V_max','voltage_min_V_min'} <= set(out.columns):
        out['voltage_window'] = out['voltage_max_V_max'] - out['voltage_min_V_min']
    if 'charge_capacity_ah_sum' in out.columns and 'discharge_capacity_ah_sum' in out.columns:
        out['q_charge_to_discharge_ratio'] = out['charge_capacity_ah_sum'] / (out['discharge_capacity_ah_sum'] + 1e-12)
    if 'charge_energy_Wh_sum' in out.columns and 'discharge_energy_Wh_sum' in out.columns:
        out['energy_charge_to_discharge_ratio'] = out['charge_energy_Wh_sum'] / (out['discharge_energy_Wh_sum'] + 1e-12)
    return out

def features_sheet3(s3: pd.DataFrame) -> pd.DataFrame:
    import os, re  # 仅保留 os、re；不要在函数内 import pandas as pd

    # 初始化与分组（修复缺失的 df/keys/out/gb）
    df = _clean_and_prepare_df(s3)
    keys = ['cycle']
    if 'barcode' in df.columns:
        keys.append('barcode')

    # 数值化与时间列处理
    for col in ['voltage_V','current_A','power_W','dQ_dV_mAh_per_V','contact_resistance_mohm','main_aux_voltage_diff_V','soc_dod_pct']:
        if col in df.columns:
            df[col] = _num(df[col])
    for tcol in ['time','total_time']:
        if tcol in df.columns:
            df[tcol + '_s'] = _to_seconds(df[tcol])

    # 分组与基础输出
    gb = df.groupby(keys, dropna=False)
    out = df[keys].drop_duplicates().copy()

    # 原始指标聚合
    target_cols = ['voltage_V','current_A','power_W','dQ_dV_mAh_per_V','contact_resistance_mohm','main_aux_voltage_diff_V']
    agg_cols = [c for c in target_cols if c in df.columns]
    if agg_cols:
        agg_dict = {c: ['mean','std','min','max'] for c in agg_cols}
        agg_df = gb.agg(agg_dict)
        agg_df.columns = ['_'.join([c for c in col if c]) for c in agg_df.columns.to_flat_index()]
        agg_df = agg_df.reset_index()
        out = pd.merge(out, agg_df, on=keys, how='left')

    # SOC/DOD 范围
    if 'soc_dod_pct' in df.columns:
        soc_stats = gb['soc_dod_pct'].agg(['max', 'min']).reset_index()
        soc_stats = soc_stats.rename(columns={'max': 'soc_max', 'min': 'soc_min'})
        soc_stats['soc_range'] = soc_stats['soc_max'] - soc_stats['soc_min']
        out = pd.merge(out, soc_stats[keys + ['soc_range']], on=keys, how='left')

    # 环境参数（保持原有）
    VHI = float(os.environ.get('VDROP_VHI', '4.2'))
    VLO = float(os.environ.get('VDROP_VLO', '3.6'))
    TOL_PCT = float(os.environ.get('CONST_CURRENT_TOL_PCT', '0.15'))
    MIN_DIS_I = float(os.environ.get('DISCHARGE_CURRENT_MIN', '0.1'))
    MIN_CHG_I = float(os.environ.get('CHARGE_CURRENT_MIN', '0.1'))
    IC_WIN_V = float(os.environ.get('IC_WIN_V', '0.05'))

    # 新增：Savitzky–Golay 平滑参数（可通过环境变量调节）
    SGF_WINDOW = int(os.environ.get('SGF_WINDOW', '11'))
    SGF_POLY = int(os.environ.get('SGF_POLY', '3'))
    SGF_MODE = os.environ.get('SGF_MODE', 'interp')

    # 缺失的工具函数：灵活列名匹配
    def _find_col_flexible(df: pd.DataFrame, patterns):
        cols = list(df.columns)
        # 精确匹配
        for pattern in patterns:
            if pattern in cols:
                return pattern
        # 不区分大小写匹配
        lower_cols = {c.lower(): c for c in cols}
        for pattern in patterns:
            pl = pattern.lower()
            if pl in lower_cols:
                return lower_cols[pl]
        # 包含匹配（模式包含列名）
        for pattern in patterns:
            pl = pattern.lower()
            for cl, orig in lower_cols.items():
                if pl in cl:
                    return orig
        # 反向包含匹配（列名包含模式）
        for pattern in patterns:
            pl = pattern.lower()
            for cl, orig in lower_cols.items():
                if cl in pl:
                    return orig
        return None

    # 平滑与特征计算（保持原有）
    def _savgol_safe(y: pd.Series, win: int, poly: int, mode: str = 'interp') -> np.ndarray:
        """对一维序列做 SG 平滑，自动修正窗口与阶数，SciPy 不可用时回退到中心滑动平均。"""
        vals = pd.to_numeric(y, errors='coerce').astype(float)
        n = int(vals.notna().sum())
        if n < 5:
            return vals.values  # 点太少，直接返回

        # 确保窗口为奇数且不超过有效点数
        win = max(3, win)
        if win % 2 == 0:
            win -= 1
        win = min(win, n if n % 2 == 1 else n - 1)
        poly = max(1, min(poly, win - 1))

        # 填充 NaN 以避免滤波失败
        filled = vals.copy().interpolate(limit_direction='both')
        filled = filled.fillna(method='bfill').fillna(method='ffill')

        try:
            from scipy.signal import savgol_filter
            return savgol_filter(filled.values, window_length=win, polyorder=poly, mode=mode)
        except Exception:
            # 回退：中心滑动平均
            return pd.Series(filled).rolling(win, center=True, min_periods=1).mean().values

    def _interp_cross(x0: float, y0: float, x1: float, y1: float, thr: float) -> float:
        """在线性段 (x0,y0)-(x1,y1) 上求 y=thr 的交点 x。"""
        if y1 == y0:
            return x0
        return x0 + (thr - y0) * (x1 - x0) / (y1 - y0)

    def _cycle_factors(g: pd.DataFrame) -> pd.Series:
        """
        计算健康因子，使用灵活的列名匹配
        """
        # 使用灵活匹配查找列名
        voltage_patterns = ['voltage_V','Voltage','V','voltage','电压(V)','电压']
        current_patterns = ['current_A','Current','I','current','电流(A)','电流']
        dqdv_patterns = ['dQ_dV_mAh_per_V','dQ/dV','dqdv','DQDV','IC','ic']
        time_patterns = ['total_time_s','time_s','Time','time','TIME','总时间(hh:mm:ss)_s','时间(hh:mm:ss)_s']
        
        v_col = _find_col_flexible(g, voltage_patterns)
        i_col = _find_col_flexible(g, current_patterns)
        dqdv_col = _find_col_flexible(g, dqdv_patterns)
        
        # 查找时间列（优先已转换的_s列）
        t_col = None
        for pattern in time_patterns:
            col = _find_col_flexible(g, [pattern])
            if col:
                t_col = col
                break
        
        # 如果缺少关键列，返回NaN值的健康因子
        if not v_col or not i_col:
            return pd.Series({
                'equal_vdrop_time_s': np.nan,
                'equal_vdrop_slope_V_per_s': np.nan,
                'discharge_avgV': np.nan,
                'ic_peak_voltage': np.nan,
                'ic_peak_height': np.nan,
                'ic_halfwidth_V': np.nan,
                'ic_area_win': np.nan
            })
        
        # 获取实际数据
        v = g[v_col] if v_col else None
        i = g[i_col] if i_col else None
        dqdv = g[dqdv_col] if dqdv_col else None
        t = g[t_col] if t_col else None

        dis_mask = None
        if i is not None:
            i_num = pd.to_numeric(i, errors='coerce')
            dis_mask = i_num < -MIN_DIS_I
        charge_mask = None
        if i is not None:
            i_num = pd.to_numeric(i, errors='coerce')
            charge_mask = i_num > MIN_CHG_I
        discharge_avgV = np.nan
        if v is not None and dis_mask is not None and dis_mask.any():
            discharge_avgV = float(pd.to_numeric(v[dis_mask], errors='coerce').dropna().mean())

        equal_vdrop_time_s = np.nan
        equal_vdrop_slope_V_per_s = np.nan
        if v is not None and t is not None and dis_mask is not None and dis_mask.any():
            vg = pd.to_numeric(v[dis_mask], errors='coerce')
            tg = pd.to_numeric(t[dis_mask], errors='coerce')

            const_mask = None
            if i is not None:
                ig = pd.to_numeric(i[dis_mask], errors='coerce').abs()
                med = float(np.nanmedian(ig))
                if med > 0:
                    tol = TOL_PCT * med
                    const_mask = (np.abs(ig - med) <= tol)
            if const_mask is not None and const_mask.any():
                idx_const = vg.index[const_mask]
                vg = vg.loc[idx_const]
                tg = tg.loc[idx_const]

            order = tg.argsort(kind='mergesort')
            vg = vg.iloc[order]
            tg = tg.iloc[order]

            try:
                i_hi = int(np.where(vg.values <= VHI)[0][0])
                i_lo = int(np.where(vg.values <= VLO)[0][0])
                if i_lo > i_hi:
                    equal_vdrop_time_s = float(tg.values[i_lo] - tg.values[i_hi])
                    # 取区间内的线性拟合斜率（dV/dt）
                    vv_seg = vg.values[i_hi:i_lo+1]
                    tt_seg = tg.values[i_hi:i_lo+1]
                    if len(vv_seg) >= 3:
                        m, b = np.polyfit(tt_seg, vv_seg, 1)
                        equal_vdrop_slope_V_per_s = float(m)
            except Exception:
                pass

        ic_peak_voltage = np.nan
        ic_peak_height = np.nan
        ic_halfwidth_V = np.nan
        ic_area_win = np.nan
        if dqdv is not None and v is not None and dis_mask is not None and dis_mask.any():
            d = pd.to_numeric(dqdv[dis_mask], errors='coerce')
            vv = pd.to_numeric(v[dis_mask], errors='coerce')
            if d.notna().any() and vv.notna().any():
                idx_peak = int(d.abs().idxmax())
                peak_h = float(d.loc[idx_peak]) if idx_peak in d.index else float(d.abs().max())
                peak_v = float(vv.loc[idx_peak]) if idx_peak in vv.index else np.nan
                ic_peak_height = peak_h
                ic_peak_voltage = peak_v
                # 半高宽（按 |dQ/dV| ≥ 0.5*峰高 的电压跨度）
                try:
                    thr = 0.5 * abs(peak_h)
                    mask_half = d.abs() >= thr
                    if mask_half.any():
                        v_half = vv[mask_half]
                        ic_halfwidth_V = float(v_half.max() - v_half.min())
                except Exception:
                    pass
                # 峰窗面积（±IC_WIN_V）
                try:
                    m_win = (vv >= (peak_v - IC_WIN_V)) & (vv <= (peak_v + IC_WIN_V))
                    if m_win.any():
                        # 简化为矩形近似的积分（也可用梯形积分）
                        ic_area_win = float((d.abs()[m_win]).sum())
                except Exception:
                    pass

        return pd.Series({
            'equal_vdrop_time_s': equal_vdrop_time_s,
            'equal_vdrop_slope_V_per_s': equal_vdrop_slope_V_per_s,
            'discharge_avgV': discharge_avgV,
            'ic_peak_voltage': ic_peak_voltage,
            'ic_peak_height': ic_peak_height,
            'ic_halfwidth_V': ic_halfwidth_V,
            'ic_area_win': ic_area_win
        })

    hf = gb.apply(_cycle_factors).reset_index()
    out = pd.merge(out, hf, on=keys, how='left')

    # 温度列聚合（英文列名）
    temp_patterns = ['Temp', 'Temperature', 'thermal']
    temp_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        for pattern in temp_patterns:
            if pattern.lower() in col_lower:
                temp_cols.append(col)
                break
    
    if temp_cols:
        pass
        for c in temp_cols:
            df[c] = _num(df[c])
        agg_dict_temp = {c: ['mean','std','min','max'] for c in temp_cols}
        temp_df = gb.agg(agg_dict_temp)
        temp_df.columns = ['_'.join([c for c in col if c]) for col in temp_df.columns.to_flat_index()]
        temp_df = temp_df.reset_index()
        out = pd.merge(out, temp_df, on=keys, how='left')
    else:
        pass

    return out

def features_ceemdan_capacity(s1: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    df = _clean_and_prepare_df(s1)
    keys = ['cycle']
    if 'barcode' in df.columns:
        keys.append('barcode')
    out = df[keys].drop_duplicates().copy()
    cap_col = _find_col(df, ['discharge_capacity_ah','capacity_ah','Discharge Capacity','Capacity(Ah)'])
    if cap_col is None:
        return out
    df[cap_col] = pd.to_numeric(df[cap_col], errors='coerce')
    if 'barcode' in df.columns:
        rows = []
        for bc, g in df.groupby('barcode', sort=False):
            g = g.sort_values('cycle')
            s = pd.Series(g[cap_col])
            miss = s.isna().to_numpy()
            miss_ratio = float(np.mean(miss)) if len(miss) > 0 else 0.0
            consec = 0
            max_consec = 0
            for v in miss:
                consec = consec + 1 if v else 0
                if consec > max_consec:
                    max_consec = consec
            s2 = s.interpolate(method='linear', limit_area='inside')
            if pd.isna(s2.iloc[0]):
                idx = s2.first_valid_index()
                if idx is not None and idx <= 3:
                    s2.iloc[:idx] = s2.iloc[idx]
            if pd.isna(s2.iloc[-1]):
                idx = s2.last_valid_index()
                if idx is not None:
                    tail_len = len(s2) - 1 - int(idx)
                    if tail_len <= 3:
                        s2.iloc[int(idx)+1:] = s2.iloc[int(idx)]
            if s2.isna().any():
                val = float(np.nanmedian(s2.values)) if np.isfinite(float(np.nanmedian(s2.values))) else 0.0
                s2 = s2.fillna(val)
            x = s2.values
            try:
                from PyEMD import CEEMDAN
                ce = CEEMDAN()
                imfs = ce.ceemdan(x)
                # 根据 IMFs 组装低频/高频/残差（保留你现有逻辑）
                # low, high, residual = ...
            except Exception:
                import numpy as np
                L = len(x)
                if L < 10:
                    # 极短序列保护：低频=原序列，高频/残差=0
                    low = x.copy()
                    high = np.zeros_like(x)
                    residual = np.zeros_like(x)
                else:
                    # 自适应窗口：至少5点，至多长度1/3，不超过15，强制奇数
                    w = max(5, min(L // 3, 15))
                    if w % 2 == 0:
                        w += 1
                    try:
                        from scipy.signal import savgol_filter
                        low = savgol_filter(x, window_length=w, polyorder=2, mode='interp')
                    except Exception:
                        # 若无 scipy 或失败则回退到中心滚动平均
                        low = pd.Series(x).rolling(window=w, min_periods=1, center=True).mean().values
                    high = x - low
                    residual = np.zeros_like(x)
            # 使用 low/high/residual 继续你后续的特征聚合与输出
            rows.append(pd.DataFrame({
                'cycle': g['cycle'].values,
                'barcode': bc,
                'ceemdan_imf_high_sum': high,
                'ceemdan_imf_low_sum': low,
                'ceemdan_residual': residual,
                'capacity_missing_ratio': [miss_ratio] * len(g),
                'capacity_max_consec_missing': [int(max_consec)] * len(g)
            }))
        return pd.concat(rows, ignore_index=True)
    else:
        g = df.sort_values('cycle')
        s = pd.Series(g[cap_col])
        miss = s.isna().to_numpy()
        miss_ratio = float(np.mean(miss)) if len(miss) > 0 else 0.0
        consec = 0
        max_consec = 0
        for v in miss:
            consec = consec + 1 if v else 0
            if consec > max_consec:
                max_consec = consec
        s2 = s.interpolate(method='linear', limit_area='inside')
        if pd.isna(s2.iloc[0]):
            idx = s2.first_valid_index()
            if idx is not None and idx <= 3:
                s2.iloc[:idx] = s2.iloc[idx]
        if pd.isna(s2.iloc[-1]):
            idx = s2.last_valid_index()
            if idx is not None:
                tail_len = len(s2) - 1 - int(idx)
                if tail_len <= 3:
                    s2.iloc[int(idx)+1:] = s2.iloc[int(idx)]
        if s2.isna().any():
            val = float(np.nanmedian(s2.values)) if np.isfinite(float(np.nanmedian(s2.values))) else 0.0
            s2 = s2.fillna(val)
        x = s2.values
        L = len(x)
        if L < 10:
            # 太短，直接返回原序列
            high = np.zeros_like(x)
            low = x.copy()
            residual = np.zeros_like(x)
        else:
            try:
                from PyEMD import CEEMDAN
                ce = CEEMDAN()
                imfs = ce.ceemdan(x)
                n = imfs.shape[0] if imfs.ndim == 2 else 0
                high = imfs[:min(2, n), :].sum(axis=0) if n > 0 else np.full_like(x, np.nan, dtype=float)
                low = imfs[max(n-2, 0):, :].sum(axis=0) if n > 0 else np.full_like(x, np.nan, dtype=float)
                residual = x - (imfs.sum(axis=0) if n > 0 else np.zeros_like(x))
            except Exception:
                # 自适应窗口：至少5个点，至多序列长度的1/3，保证奇数
                w = max(5, min(L // 3, 15))
                try:
                    from scipy.signal import savgol_filter
                    low = savgol_filter(x, window_length=(w | 1), polyorder=2)
                except Exception:
                    low = pd.Series(x).rolling(window=w, min_periods=1, center=True).mean().values
                high = x - low
                residual = np.zeros_like(x)
        return pd.DataFrame({
            'cycle': g['cycle'].values,
            'ceemdan_imf_high_sum': high,
            'ceemdan_imf_low_sum': low,
            'ceemdan_residual': residual,
            'capacity_missing_ratio': [miss_ratio] * len(g),
            'capacity_max_consec_missing': [int(max_consec)] * len(g)
        })

def merge_feature_tables(f1: pd.DataFrame, f2: pd.DataFrame, f3: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ['cycle','barcode'] if k in f1.columns or k in f2.columns or k in f3.columns]
    if not keys:
        keys = ['cycle']
    out = pd.merge(f1, f2, on=keys, how='outer')
    out = pd.merge(out, f3, on=keys, how='outer')
    out = out.sort_values(keys)
    return out
