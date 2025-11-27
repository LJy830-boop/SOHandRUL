import os
import pandas as pd
import matplotlib.pyplot as plt

def resolve_artifacts_dir():
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, 'artifacts'),
        os.path.join(os.path.dirname(cwd), 'artifacts'),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts')),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return os.path.join(cwd, 'artifacts')

def _load_top_csv(art_dir, fname):
    path = os.path.join(art_dir, fname)
    if not os.path.isfile(path):
        print(f'未找到 {path}')
        return None
    df = pd.read_csv(path)
    if 'correlation' in df.columns:
        df = df.reindex(df['correlation'].abs().sort_values(ascending=False).index)
    return df.head(20)

def _label_series(df):
    # 优先用 original_name；如果没有或为空则用 feature
    if df is None:
        return None
    if 'original_name' in df.columns and df['original_name'].notna().any():
        labs = df['original_name'].fillna(df['feature']).astype(str)
    else:
        labs = df['feature'].astype(str)
    return labs

def _plot_bar(df, title, save_path):
    if df is None or df.empty:
        print(f'{title}: 数据为空，跳过')
        return
    labels = _label_series(df)
    corr = df['correlation'].astype(float)
    colors = ['#c62828' if v < 0 else '#1565c0' for v in corr]  # 负为红、正为蓝
    plt.figure(figsize=(12, 7))
    # 横向条形图更适合长标签
    y_pos = range(len(labels))
    plt.barh(y_pos, corr, color=colors)
    plt.yticks(y_pos, labels)
    plt.xlabel('Spearman correlation')
    plt.title(title)
    # 在条末标注数值
    for i, v in enumerate(corr):
        plt.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}',
                 va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {save_path}')

if __name__ == '__main__':
    art = resolve_artifacts_dir()
    # 优先用真实列名的 Top-20 文件；不存在则使用原始 CSV
    pred_df = _load_top_csv(art, 'top20_feature_pred_names.csv') or _load_top_csv(art, 'feature_vs_pred_corr.csv')
    true_df = _load_top_csv(art, 'top20_feature_true_names.csv') or _load_top_csv(art, 'feature_vs_true_corr.csv')
    _plot_bar(pred_df, 'Top-20 Feature vs Prediction (Spearman)', os.path.join(art, 'top20_pred_corr_bar.png'))
    _plot_bar(true_df, 'Top-20 Feature vs True (Spearman)', os.path.join(art, 'top20_true_corr_bar.png'))