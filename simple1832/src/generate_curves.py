#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从by_battery目录重构完整的capacity_retention_curve.csv和rul_curve.csv
"""

import pandas as pd
import os
import glob
import numpy as np

def reconstruct_soh_curve(base_dir):
    """重构SOH的capacity_retention_curve.csv"""
    print("\n=== 开始重构SOH capacity_retention_curve.csv ===")
    
    soh_dir = os.path.join(base_dir, 'soh', 'by_battery')
    soh_files = glob.glob(os.path.join(soh_dir, '*_forecast_capacity_overlay.csv'))
    
    print(f"找到{len(soh_files)}个SOH文件")
    
    all_data = []
    
    for f in sorted(soh_files):
        battery_id = os.path.basename(f).replace('group_', '').replace('.csv_forecast_capacity_overlay.csv', '')
        df = pd.read_csv(f, encoding='utf-8-sig')
        
        print(f"\n处理 {battery_id}: {len(df)}行, 列: {list(df.columns)}")
        
        if len(df) == 0:
            continue
        
        # 提取数据
        base_cap = None
        for idx, row in df.iterrows():
            cycle = row.get('cycle', np.nan)
            cap_true_ah = row.get('cap_true_ah', np.nan)
            cap_stacking = row.get('cap_pred_ah_stacking', np.nan)
            cap_blending = row.get('cap_pred_ah_blending', np.nan)
            
            # 设置基准容量（第一个有效的真实容量）
            if base_cap is None and not np.isnan(cap_true_ah) and cap_true_ah > 0:
                base_cap = cap_true_ah
                print(f"  基准容量: {base_cap:.4f} Ah")
            
            if not np.isnan(cap_true_ah) and base_cap and base_cap > 0:
                retention_true = (cap_true_ah / base_cap) * 100
                retention_stacking = (cap_stacking / base_cap) * 100 if not np.isnan(cap_stacking) else np.nan
                retention_blending = (cap_blending / base_cap) * 100 if not np.isnan(cap_blending) else np.nan
                
                all_data.append({
                    'barcode': battery_id,
                    'cycle': cycle,
                    'capacity_retention_pct_stacking': retention_stacking,
                    'capacity_retention_pct_blending': retention_blending,
                    'capacity_retention_pct_true': retention_true
                })
    
    if len(all_data) > 0:
        df_result = pd.DataFrame(all_data)
        output_path = os.path.join(base_dir, 'soh', 'capacity_retention_curve.csv')
        df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ SOH capacity_retention_curve.csv已生成！")
        print(f"   路径: {output_path}")
        print(f"   总行数: {len(df_result)}")
        print(f"   电池数量: {df_result['barcode'].nunique()}")
        print(f"\n前10行预览：")
        print(df_result.head(10))
        
        # 统计每个电池的预测值覆盖率
        print(f"\n预测值覆盖率统计：")
        for bat in df_result['barcode'].unique():
            df_bat = df_result[df_result['barcode'] == bat]
            stacking_valid = df_bat['capacity_retention_pct_stacking'].notna().sum()
            blending_valid = df_bat['capacity_retention_pct_blending'].notna().sum()
            total = len(df_bat)
            print(f"  {bat}: stacking {stacking_valid}/{total} ({stacking_valid/total*100:.1f}%), blending {blending_valid}/{total} ({blending_valid/total*100:.1f}%)")
        
        return True
    else:
        print("❌ 没有找到有效数据！")
        return False

def reconstruct_rul_curve(base_dir):
    """重构RUL的rul_curve.csv"""
    print("\n\n=== 开始重构RUL rul_curve.csv ===")
    
    rul_dir = os.path.join(base_dir, 'rul', 'by_battery')
    rul_files = glob.glob(os.path.join(rul_dir, '*_forecast_capacity_overlay.csv'))
    
    print(f"找到{len(rul_files)}个RUL文件")
    
    all_data = []
    
    for f in sorted(rul_files):
        battery_id = os.path.basename(f).replace('group_', '').replace('.csv_forecast_capacity_overlay.csv', '')
        df = pd.read_csv(f, encoding='utf-8-sig')
        
        print(f"\n处理 {battery_id}: {len(df)}行, 列: {list(df.columns)}")
        
        if len(df) == 0:
            continue
        
        # 提取数据
        base_cap = None
        for idx, row in df.iterrows():
            cycle = row.get('cycle', np.nan)
            cap_true_ah = row.get('cap_true_ah', np.nan)
            cap_stacking = row.get('cap_pred_ah_stacking', np.nan)
            cap_blending = row.get('cap_pred_ah_blending', np.nan)
            
            # 设置基准容量（第一个有效的真实容量）
            if base_cap is None and not np.isnan(cap_true_ah) and cap_true_ah > 0:
                base_cap = cap_true_ah
                print(f"  基准容量: {base_cap:.4f} Ah")
            
            if not np.isnan(cap_true_ah) and base_cap and base_cap > 0:
                retention_true = (cap_true_ah / base_cap) * 100
                retention_stacking = (cap_stacking / base_cap) * 100 if not np.isnan(cap_stacking) else np.nan
                retention_blending = (cap_blending / base_cap) * 100 if not np.isnan(cap_blending) else np.nan
                
                all_data.append({
                    'barcode': battery_id,
                    'cycle': cycle,
                    'capacity_retention_pct_stacking': retention_stacking,
                    'capacity_retention_pct_blending': retention_blending,
                    'capacity_retention_pct_true': retention_true
                })
    
    if len(all_data) > 0:
        df_result = pd.DataFrame(all_data)
        output_path = os.path.join(base_dir, 'rul', 'rul_curve.csv')
        df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ RUL rul_curve.csv已生成！")
        print(f"   路径: {output_path}")
        print(f"   总行数: {len(df_result)}")
        print(f"   电池数量: {df_result['barcode'].nunique()}")
        print(f"\n前10行预览：")
        print(df_result.head(10))
        
        # 统计每个电池的预测值覆盖率
        print(f"\n预测值覆盖率统计：")
        for bat in df_result['barcode'].unique():
            df_bat = df_result[df_result['barcode'] == bat]
            stacking_valid = df_bat['capacity_retention_pct_stacking'].notna().sum()
            blending_valid = df_bat['capacity_retention_pct_blending'].notna().sum()
            total = len(df_bat)
            print(f"  {bat}: stacking {stacking_valid}/{total} ({stacking_valid/total*100:.1f}%), blending {blending_valid}/{total} ({blending_valid/total*100:.1f}%)")
        
        return True
    else:
        print("❌ 没有找到有效数据！")
        return False

if __name__ == '__main__':
    base_dir = r'c:\Users\LJy830\Desktop\博士论文\Battery\ceshi\SOHANDRUL\artifacts'
    
    soh_ok = reconstruct_soh_curve(base_dir)
    rul_ok = reconstruct_rul_curve(base_dir)
    
    print("\n\n=== 总结 ===")
    print(f"SOH任务: {'✅ 成功' if soh_ok else '❌ 失败'}")
    print(f"RUL任务: {'✅ 成功' if rul_ok else '❌ 失败'}")
    print("\n所有任务完成！")
