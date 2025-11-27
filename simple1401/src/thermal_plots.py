import os
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
sns.set(style="white", context="talk")

def to_num(s):
    return pd.to_numeric(s, errors='coerce')

def find_cols_by_patterns(df, patterns):
    pats = [p.lower() for p in patterns]
    out = []
    for c in df.columns:
        name = str(c).lower()
        if any(p in name for p in pats):
            out.append(c)
    return out

def find_temperature_cols(df):
    # 中文/英文常见关键词，必要时可根据你的列名再补充
    patterns = [
        '温度', '壳温', '箱温', '机壳', '电芯温度', '芯温',
        'temperature', 'temp', 'coolant', 'thermal'
    ]
    cols = find_cols_by_patterns(df, patterns)
    # 避免把“环境温度补偿”等非数值列误选进来（如是文本列）
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]
    return cols

def find_target_col(df):
    aliases = [
        'SOH', 'state of health', 'health',
        '容量保持率', '容量保持率(%)', 'capacity retention', 'remaining capacity ratio'
    ]
    hits = find_cols_by_patterns(df, aliases)
    return hits[0] if hits else None

def find_cycle_col(df):
    aliases = ['循环号', 'cycle', '循环', 'Cycle']
    hits = find_cols_by_patterns(df, aliases)
    return hits[0] if hits else None

def corr_and_p(x, y, method='spearman'):
    x = to_num(x)
    y = to_num(y)
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    if method == 'pearson':
        from scipy.stats import pearsonr
        r, p = pearsonr(x[mask], y[mask])
    else:
        r, p = spearmanr(x[mask], y[mask])
    return r, p, int(mask.sum())

def corr_matrix(df, cols, method='spearman'):
    n = len(cols)
    mat = np.full((n, n), np.nan)
    pvals = np.full((n, n), np.nan)
    ns = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            r, p, k = corr_and_p(df[cols[i]], df[cols[j]], method)
            mat[i, j] = r
            pvals[i, j] = p
            ns[i, j] = k
    return (
        pd.DataFrame(mat, index=cols, columns=cols),
        pd.DataFrame(pvals, index=cols, columns=cols),
        pd.DataFrame(ns, index=cols, columns=cols),
    )

def corr_to_target(df, temp_cols, target_col, method='spearman'):
    rows = []
    for c in temp_cols:
        r, p, n = corr_and_p(df[c], df[target_col], method)
        rows.append({'feature': c, 'corr': r, 'p': p, 'n': n})
    return pd.DataFrame(rows).set_index('feature')

def p_to_stars(p):
    if pd.isna(p):
        return ''
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 5e-2 else ''

def plot_corr_heatmap(corr_df, p_df=None, title='热相关矩阵', mask_upper=True):
    annot = None
    if p_df is not None:
        annot = corr_df.copy().astype(object)
        for i in range(annot.shape[0]):
            for j in range(annot.shape[1]):
                val = corr_df.iat[i, j]
                pv = p_df.iat[i, j]
                annot.iat[i, j] = f"{val:.2f}{p_to_stars(pv)}" if not pd.isna(val) else ''
    mask = np.triu(np.ones_like(corr_df, dtype=bool)) if mask_upper else None
    plt.figure(figsize=(max(8, 0.6 * corr_df.shape[1] + 2), max(8, 0.6 * corr_df.shape[0] + 2)))
    ax = sns.heatmap(corr_df, cmap='coolwarm', vmin=-1, vmax=1, annot=annot, fmt='', linewidths=0.5, linecolor='white', mask=mask)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def bin_label(start, width):
    return f"{start+1}-{start+width}"

def bin_corr_heatmap(df, temp_cols, target_col, cycle_col, method='spearman', window=50, min_n=5):
    cycles = to_num(df[cycle_col])
    df = df.copy()
    df['_bin'] = ((cycles - 1) // window).astype('Int64')
    bins = [b for b in sorted(df['_bin'].dropna().unique())]
    if not bins:
        raise ValueError("无法生成循环窗口，请检查循环号列是否为数字。")
    # 计算每个窗口内的温度列与目标的相关性
    heat = pd.DataFrame(index=temp_cols, columns=[bin_label(int(b) * window, window) for b in bins], dtype=float)
    for b in bins:
        sub = df[df['_bin'] == b]
        for c in temp_cols:
            r, p, n = corr_and_p(sub[c], sub[target_col], method)
            heat.loc[c, bin_label(int(b) * window, window)] = r if n >= min_n else np.nan
    return heat

def read_excel_smart(xlsx_path, sheet=None):
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"文件不存在：{xlsx_path}")
    xls = pd.ExcelFile(xlsx_path)
    if sheet is not None:
        return pd.read_excel(xls, sheet_name=sheet)
    # 自动挑选包含温度列的表
    best_df = None
    best_score = -1
    for name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=name)
        score = len(find_temperature_cols(df))
        if score > best_score:
            best_score = score
            best_df = df
    return best_df if best_df is not None else pd.read_excel(xls, sheet_name=xls.sheet_names[0])

def main():
    parser = argparse.ArgumentParser(description="电芯热相关图生成")
    parser.add_argument('--xlsx', type=str, default=os.path.join('data', 'battery.xlsx'), help="Excel 文件路径")
    parser.add_argument('--sheet', type=str, default=None, help="工作表名称或索引（可选）")
    parser.add_argument('--method', type=str, default='spearman', choices=['spearman', 'pearson'], help="相关系数方法")
    parser.add_argument('--window', type=int, default=50, help="按循环分窗的窗口大小")
    parser.add_argument('--aggregate', action='store_true', help="按循环号聚合（对温度取均值）")
    args = parser.parse_args()

    df = read_excel_smart(args.xlsx, sheet=args.sheet)
    temp_cols = find_temperature_cols(df)
    target_col = find_target_col(df)
    cycle_col = find_cycle_col(df)

    if not temp_cols:
        raise ValueError("未识别到任何温度相关列，请检查列名（包含“温度/Temp/Temperature”等）。")
    if target_col is None:
        warnings.warn("未识别到目标列（如“容量保持率(%)”或“SOH”），将仅生成温度之间的相关矩阵。")
    else:
        print(f"目标列：{target_col}")
    if cycle_col is None:
        warnings.warn("未识别到循环号列（如“循环号/Cycle”），将跳过分循环窗口热图。")
    else:
        print(f"循环号列：{cycle_col}")

    # 强制数值化
    for c in temp_cols:
        df[c] = to_num(df[c])
    if target_col:
        df[target_col] = to_num(df[target_col])

    # 可选按循环号聚合，得到每循环的温度均值与目标均值
    if args.aggregate and cycle_col:
        keys = [cycle_col]
        if '条码' in df.columns:
            keys.append('条码')
        num_cols = temp_cols + ([target_col] if target_col else [])
        df = df.groupby(keys, dropna=False)[num_cols].mean().reset_index()

    # 1) 温度列之间 + 可选包含目标列的总体相关矩阵
    cols_for_matrix = temp_cols + ([target_col] if target_col else [])
    corr_df, p_df, _ = corr_matrix(df, cols_for_matrix, method=args.method)
    plot_corr_heatmap(corr_df, p_df if target_col else None, title='温度-目标总体相关热力图')

    # 2) 按循环窗口的温度-目标相关演化热图
    if target_col and cycle_col:
        heat = bin_corr_heatmap(df, temp_cols, target_col, cycle_col, method=args.method, window=args.window)
        plt.figure(figsize=(max(8, 0.5 * heat.shape[1] + 2), max(8, 0.5 * heat.shape[0] + 2)))
        sns.heatmap(heat, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt='.2f', linewidths=0.5, linecolor='white')
        plt.title(f'温度-目标分循环窗口相关热力图（窗口={args.window}）')
        plt.xlabel('循环窗口')
        plt.ylabel('温度传感器/温度特征')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()