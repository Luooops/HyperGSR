import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _autolabel(ax, rects, fontsize=9, fmt="{:.3f}"):
    """在柱子上方标注数值"""
    for r in rects:
        h = r.get_height()
        ax.annotate(fmt.format(h),
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=fontsize, rotation=0)

def _ensure_numeric(df):
    """把除 'fold' 外的列转成 float，避免字符串类型影响绘图/计算"""
    for c in df.columns:
        if c != "fold":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def plot_metrics_compare(
    csv_a: str,
    csv_b: str,
    labels=("Model A", "Model B"),
    out_dir: str = "./plots",
    prefix: str = "compare"
):
    """
    读取两份评估 CSV，分别为 fold_1, fold_2, fold_3, average 生成 4 张图。
    每张图内：
      - 上：分组柱状图（0~1 量级指标）
      - 下：单独的 laplacian_frobenius_distance
    参数:
      csv_a, csv_b: 两份 CSV 路径（列名需一致，含 'fold'）
      labels: 两条曲线/两组柱子的图例名 (csv_a 对应 labels[0], csv_b 对应 labels[1])
      out_dir: 图片输出目录
      prefix: 文件名前缀
    """
    os.makedirs(out_dir, exist_ok=True)

    df_a = _ensure_numeric(pd.read_csv(csv_a))
    df_b = _ensure_numeric(pd.read_csv(csv_b))

    # 统一用两份 CSV 的列交集，保证对齐
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    if "fold" not in common_cols:
        raise ValueError("Both CSVs must contain a 'fold' column.")
    # 指标列（去掉 fold）
    metric_cols = [c for c in common_cols if c != "fold"]

    # 小量级指标与大尺度指标拆开
    small_metrics = [c for c in metric_cols if c != "laplacian_frobenius_distance"]
    big_metric    = "laplacian_frobenius_distance" if "laplacian_frobenius_distance" in metric_cols else None

    # 需要绘制的行标签（fold_1, fold_2, fold_3, average）
    # 以两份 CSV 交集为准，且按约定顺序排序
    order = ["fold_1", "fold_2", "fold_3", "average"]
    rows_a = set(df_a["fold"].tolist())
    rows_b = set(df_b["fold"].tolist())
    rows = [r for r in order if r in rows_a and r in rows_b]
    if not rows:
        raise ValueError("No common folds found between the two CSVs.")

    # 统一风格
    plt.style.use("seaborn-v0_8-whitegrid")

    for row in rows:
        rec_a = df_a[df_a["fold"] == row]
        rec_b = df_b[df_b["fold"] == row]
        if rec_a.empty or rec_b.empty:
            continue

        # 取当前行的指标数值
        vals_a_small = [float(rec_a[m].values[0]) for m in small_metrics]
        vals_b_small = [float(rec_b[m].values[0]) for m in small_metrics]
        val_a_big = float(rec_a[big_metric].values[0]) if big_metric else None
        val_b_big = float(rec_b[big_metric].values[0]) if big_metric else None

        # 创建画布（上下两个子图）
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0]) if big_metric else None

        # ---------- 顶部：分组柱状图（小量级指标） ----------
        x = np.arange(len(small_metrics))
        width = 0.38

        bars_a = ax_top.bar(x - width/2, vals_a_small, width, label=labels[0])
        bars_b = ax_top.bar(x + width/2, vals_b_small, width, label=labels[1])

        ax_top.set_title(f"{row} — Metrics Comparison", fontsize=16, pad=12)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(small_metrics, rotation=30, ha="right", fontsize=10)
        ax_top.set_ylabel("Score", fontsize=12)
        ax_top.legend(loc="upper right", frameon=True)
        ax_top.grid(True, axis="y", linestyle="--", alpha=0.5)

        # 自动标注数值
        _autolabel(ax_top, bars_a, fontsize=8)
        _autolabel(ax_top, bars_b, fontsize=8)

        # ---------- 底部：大尺度指标（拉普拉斯弗罗贝尼乌斯距离） ----------
        if big_metric and ax_bot is not None:
            idx = np.arange(1)  # 单独一个指标
            w2 = 0.35
            bars_a2 = ax_bot.bar(idx - w2/2, [val_a_big], w2, label=labels[0])
            bars_b2 = ax_bot.bar(idx + w2/2, [val_b_big], w2, label=labels[1])

            ax_bot.set_title(f"{big_metric}", fontsize=14, pad=8)
            ax_bot.set_xticks(idx)
            ax_bot.set_xticklabels([big_metric], rotation=0, fontsize=10)
            ax_bot.set_ylabel("Distance", fontsize=12)
            ax_bot.grid(True, axis="y", linestyle="--", alpha=0.5)

            _autolabel(ax_bot, bars_a2, fontsize=8, fmt="{:.2f}")
            _autolabel(ax_bot, bars_b2, fontsize=8, fmt="{:.2f}")

        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{prefix}_{row}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# -------------------------
# 使用示例
if __name__ == "__main__":
    plot_metrics_compare(
        csv_a="results/stp_gsr/csv/run1/metrics.csv",
        csv_b="results/hyper_gsr/csv/trans/run_baseline/metrics.csv",
        labels=("STP-GSR", "HyperGSR(baseline)"),
        out_dir="models/STP-GSR/model_results/plots_baseline",
        prefix="stp_vs_hyper_baseline"
    )
