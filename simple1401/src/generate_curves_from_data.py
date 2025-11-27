#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接从data目录读取原始CSV文件，生成capacity_retention_curve.csv和rul_curve.csv
"""

import pandas as pd
import os
import glob
import numpy as np
from scipy.signal import savgol_filter

def process_battery_data(data_dir, output_base_dir):
    """
    处理data目录下的所有电池CSV文件
    """
    print("\n=== 开始处理原始电池数据 ===")
    
    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"找到{len(csv_files)}个电池数据文件")
    
    all_capacity_data = []
    all_rul_data = []
    
    for csv_file in sorted(csv_files):
        battery_id = os.path.basename(csv_file).replace('.csv', '')
        print(f"\n处理 {battery_id}...")
        
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except:
            try:
                df = pd.read_csv(csv_file, encoding='gbk')
            except:
                print(f"  ⚠️ 无法读取文件，跳过")
                continue
        
        # 查找容量列
        cap_col = None
        for col_name in ['放电容量(Ah)', 'discharge_capacity_ah', 'Discharge Capacity', 'Capacity(Ah)', '容量']:
            if col_name in df.columns:
                cap_col = col_name
                break
        
        if cap_col is None:
            print(f"  ⚠️ 未找到容量列，跳过")
            continue
        
        # 查找循环号列
        cycle_col = None
        for col_name in ['循环号', 'cycle', 'Cycle', 'Cycle No']:
            if col_name in df.columns:
                cycle_col = col_name
                break
        
        if cycle_col is None:
            # 使用行号作为cycle
            df['cycle'] = range(1, len(df) + 1)
            cycle_col = 'cycle'
        
        # 提取有效数据
        df['cycle_num'] = pd.to_numeric(df[cycle_col], errors='coerce')
        df['capacity'] = pd.to_numeric(df[cap_col], errors='coerce')
        df = df.dropna(subset=['cycle_num', 'capacity'])
        df = df[df['capacity'] > 0]
        df = df.sort_values('cycle_num')
        
        if len(df) < 5:
            print(f"  ⚠️ 有效数据不足，跳过")
            continue
        
        # 计算容量保持率
        base_capacity = df['capacity'].iloc[0]
        df['capacity_retention_pct'] = (df['capacity'] / base_capacity) * 100
        
        print(f"  基准容量: {base_capacity:.4f} Ah")
        print(f"  有效数据: {len(df)} 个cycle")
        
        # 异常值检测和平滑
        retention_vals = df['capacity_retention_pct'].values
        retention_cleaned = retention_vals.copy()
        
        if len(retention_vals) >= 7:
            rolling_median = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).median()
            rolling_std = pd.Series(retention_vals).rolling(window=7, min_periods=3, center=True).std()
            outlier_mask = np.abs(retention_vals - rolling_median) > (3 * rolling_std)
            outlier_mask = outlier_mask.fillna(False).values
            
            if outlier_mask.sum() > 0:
                print(f"  🔍 检测到 {outlier_mask.sum()} 个异常值")
                outlier_indices = np.where(outlier_mask)[0]
                for idx in outlier_indices:
                    cycle_num = df['cycle_num'].iloc[idx]
                    old_val = retention_vals[idx]
                    new_val = rolling_median.iloc[idx]
                    print(f"     Cycle {int(cycle_num)}: {old_val:.2f}% → {new_val:.2f}%")
                retention_cleaned[outlier_mask] = rolling_median.values[outlier_mask]
        
        # 平滑处理
        if len(retention_cleaned) >= 11:
            window = min(11, len(retention_cleaned) if len(retention_cleaned) % 2 == 1 else len(retention_cleaned) - 1)
            smoothed = savgol_filter(retention_cleaned, window_length=window, polyorder=2, mode='nearest')
            smoothed = pd.Series(smoothed).rolling(window=5, min_periods=1, center=True).mean().values
        elif len(retention_cleaned) >= 5:
            window = min(5, len(retention_cleaned) if len(retention_cleaned) % 2 == 1 else len(retention_cleaned) - 1)
            smoothed = savgol_filter(retention_cleaned, window_length=window, polyorder=2, mode='nearest')
        else:
            smoothed = retention_cleaned
        
        df['capacity_retention_pct_smoothed'] = smoothed
        
        # 计算RUL
        threshold = 80.0
        below_threshold = smoothed <= threshold
        
        if below_threshold.any():
            eol_idx = np.where(below_threshold)[0][0]
        else:
            eol_idx = len(smoothed) - 1
        
        rul = np.arange(len(smoothed) - 1, -1, -1, dtype=float)
        rul = rul - (len(smoothed) - 1 - eol_idx)
        rul = np.maximum(0, rul)
        
        df['rul'] = rul
        
        eol_cycle = df['cycle_num'].iloc[eol_idx] if eol_idx < len(df) else df['cycle_num'].iloc[-1]
        print(f"  📊 EOL预测: Cycle {int(eol_cycle)}, 容量保持率: {smoothed[eol_idx]:.2f}%")
        print(f"  🔋 初始RUL: {int(rul[0])}")
        
        # 保存到结果列表
        for idx, row in df.iterrows():
            cycle = row['cycle_num']
            retention = row['capacity_retention_pct_smoothed']
            rul_val = row['rul']
            
            all_capacity_data.append({
                'battery_id': battery_id,
                'cycle': int(cycle),
                'capacity_retention_pct_true': retention,
                'capacity_retention_pct_stacking': np.nan,
                'capacity_retention_pct_blending': np.nan
            })
            
            all_rul_data.append({
                'battery_id': battery_id,
                'cycle': int(cycle),
                'capacity_retention_pct_true': retention,
                'capacity_retention_pct_stacking': np.nan,
                'capacity_retention_pct_blending': np.nan,
                'rul_true': rul_val,
                'is_prediction': False,
                'rul_pred_stacking': np.nan,
                'rul_pred_blending': np.nan
            })
    
    # 生成SOH曲线文件
    if all_capacity_data:
        df_soh = pd.DataFrame(all_capacity_data)
        soh_output_dir = os.path.join(output_base_dir, 'soh')
        os.makedirs(soh_output_dir, exist_ok=True)
        soh_output_path = os.path.join(soh_output_dir, 'capacity_retention_curve.csv')
        df_soh.to_csv(soh_output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ SOH曲线已生成: {soh_output_path}")
        print(f"   总数据点: {len(df_soh)}")
        print(f"   电池数量: {df_soh['battery_id'].nunique()}")
    
    # 生成RUL曲线文件
    if all_rul_data:
        df_rul = pd.DataFrame(all_rul_data)
        rul_output_dir = os.path.join(output_base_dir, 'rul')
        os.makedirs(rul_output_dir, exist_ok=True)
        rul_output_path = os.path.join(rul_output_dir, 'rul_curve.csv')
        df_rul.to_csv(rul_output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ RUL曲线已生成: {rul_output_path}")
        print(f"   总数据点: {len(df_rul)}")
        print(f"   电池数量: {df_rul['battery_id'].nunique()}")
    
    return True

if __name__ == '__main__':
    import sys
    
    # 自动检测路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = os.path.join(project_root, 'data')
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.join(project_root, 'artifacts')
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        sys.exit(1)
    
    process_battery_data(data_dir, output_dir)
    
    print("\n\n🎉 所有任务完成！")
