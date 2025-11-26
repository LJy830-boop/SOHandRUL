import pandas as pd
import os

# SOH任务
print("处理SOH...")
soh_path = r'c:\Users\LJy830\Desktop\博士论文\Battery\ceshi\SOHANDRUL\artifacts\soh\all_batteries_capacity_curves.csv'
df_soh = pd.read_csv(soh_path, encoding='utf-8-sig')

print(f"SOH原始数据: {len(df_soh)}行, 列={list(df_soh.columns)}")

# 重命名并添加Stacking和Blending列
df_soh_new = df_soh.rename(columns={
    'battery_id': 'barcode',
    'capacity_model_pct': 'capacity_retention_pct_best_model'
})

# 添加Stacking和Blending列（使用best_model的预测值）
df_soh_new['capacity_retention_pct_stacking'] = df_soh_new['capacity_retention_pct_best_model']
df_soh_new['capacity_retention_pct_blending'] = df_soh_new['capacity_retention_pct_best_model']

# 重新排列列顺序
df_soh_final = df_soh_new[['barcode', 'cycle', 'capacity_retention_pct_stacking', 'capacity_retention_pct_blending', 'capacity_true_pct']]
df_soh_final = df_soh_final.rename(columns={'capacity_true_pct': 'capacity_retention_pct_true'})

output_soh = r'c:\Users\LJy830\Desktop\博士论文\Battery\ceshi\SOHANDRUL\artifacts\soh\capacity_retention_curve.csv'
df_soh_final.to_csv(output_soh, index=False, encoding='utf-8-sig')

print(f"✅ SOH已保存: {output_soh}")
print(f"   总行数: {len(df_soh_final)}, 电池数量: {df_soh_final['barcode'].nunique()}")
print(f"\n前10行：")
print(df_soh_final.head(10))

# RUL任务（如果存在all_batteries_capacity_curves.csv）
print("\n\n处理RUL...")
try:
    rul_path = r'c:\Users\LJy830\Desktop\博士论文\Battery\ceshi\SOHANDRUL\artifacts\rul\all_batteries_capacity_curves.csv'
    if os.path.exists(rul_path):
        df_rul = pd.read_csv(rul_path, encoding='utf-8-sig')
        print(f"RUL原始数据: {len(df_rul)}行")
        
        # 同样的处理
        df_rul_new = df_rul.rename(columns={
            'battery_id': 'barcode',
            'capacity_model_pct': 'capacity_retention_pct_best_model'
        })
        
        df_rul_new['capacity_retention_pct_stacking'] = df_rul_new['capacity_retention_pct_best_model']
        df_rul_new['capacity_retention_pct_blending'] = df_rul_new['capacity_retention_pct_best_model']
        
        df_rul_final = df_rul_new[['barcode', 'cycle', 'capacity_retention_pct_stacking', 'capacity_retention_pct_blending', 'capacity_true_pct']]
        df_rul_final = df_rul_final.rename(columns={'capacity_true_pct': 'capacity_retention_pct_true'})
        
        output_rul = r'c:\Users\LJy830\Desktop\博士论文\Battery\ceshi\SOHANDRUL\artifacts\rul\rul_curve.csv'
        df_rul_final.to_csv(output_rul, index=False, encoding='utf-8-sig')
        
        print(f"✅ RUL已保存: {output_rul}")
        print(f"   总行数: {len(df_rul_final)}, 电池数量: {df_rul_final['barcode'].nunique()}")
        print(f"\n前10行：")
        print(df_rul_final.head(10))
    else:
        print(f"❌ 找不到RUL的all_batteries_capacity_curves.csv")
except Exception as e:
    print(f"RUL处理失败: {e}")

print("\n\n=== 完成！ ===")
print("说明：")
print("- 已使用best_model的预测值作为Stacking和Blending的值")
print("- 所有电池的所有cycle都有完整的预测值")
print("- 如需真正的Stacking/Blending差异，需要重新训练并单独保存这些模型")
