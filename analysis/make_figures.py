"""Sinh hình cho bài IDA 2027.

Ba hình, mỗi hình gánh một mục:
  fig1_head_gap.pdf      §4  độ chênh train-test tăng đơn điệu theo dung lượng head
  fig2_mcnemar.pdf       §5  cặp không phân biệt được vẫn đoán khác nhau hàng trăm ảnh
  fig3_diversity.pdf     §6  lợi ích ensemble đi theo bất đồng, không theo accuracy
"""

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
FIGDIR = HERE.parent / "Paper-IDA2027" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

SEEDS = ["seed36", "seed42"]
BACKBONES = ["efficient_net", "dense_net", "xception",
             "resnet_50", "inception_v3", "mobilenet_v3"]
NICE = {"efficient_net": "EffNetV2B0", "dense_net": "DenseNet201",
        "xception": "Xception", "resnet_50": "ResNet50",
        "inception_v3": "InceptionV3", "mobilenet_v3": "MobileNetV3S"}
HEADS = ["SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"]
HEAD_LABEL = {"SIMPLE_FC": "SIMPLE\\_FC", "GAP_BN_FC_Drop": "GAP+FC",
              "DUAL_POOLING_FC": "DUAL POOL"}

# bảng màu an toàn với người mù màu (Okabe-Ito)
C_ORANGE, C_BLUE, C_GREEN = "#E69F00", "#0072B2", "#009E73"
C_GREY = "#999999"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 150,
})


def backbone_of(m):
    for h in HEADS:
        if m.endswith("_" + h):
            return m[: -len(h) - 1]
    return m


def head_of(m):
    for h in HEADS:
        if m.endswith("_" + h):
            return h
    return "?"


def _load_dir(folder):
    out = {}
    for p in folder.glob("preds_*.npz"):
        d = np.load(p)
        out[p.stem.replace("preds_", "")] = (
            d["y_pred"].astype(int), d["y_prob"].astype(np.float64),
            d["y_true"].astype(int))
    return out


def load(seed):
    return _load_dir(HERE / f"results_online_{seed}")


def load_val(seed):
    return _load_dir(HERE / f"results_val_{seed}")


DATA = {s: load(s) for s in SEEDS}
TRAIN = defaultdict(dict)
with open(HERE / "train_acc.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        TRAIN[r["seed"]][r["model"]] = float(r["train_acc"])


# --------------------------------------------------------------- hình 1
def fig_head_gap():
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    colors = {h: c for h, c in zip(HEADS, (C_BLUE, C_GREEN, C_ORANGE))}
    markers = {h: m for h, m in zip(HEADS, ("o", "s", "^"))}

    xs = np.arange(len(BACKBONES))
    for h in HEADS:
        gaps = []
        for b in BACKBONES:
            m = f"{b}_{h}"
            te = (DATA["seed36"][m][0] == DATA["seed36"][m][2]).mean() * 100
            gaps.append(TRAIN["seed36"][m] - te)
        ax.plot(xs, gaps, marker=markers[h], color=colors[h], lw=1.1, ms=4,
                label=HEAD_LABEL[h].replace("\\_", "_"))

    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[b] for b in BACKBONES], rotation=20, ha="right")
    ax.set_ylabel("train $-$ test accuracy (points)")
    ax.set_ylim(0, 9)
    ax.grid(axis="y", lw=0.4, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.legend(title="classifier head", frameon=False, loc="upper left",
              ncol=3, columnspacing=1.0, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR / "fig1_head_gap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig1_head_gap.pdf")


# --------------------------------------------------------------- hình 2
def fig_mcnemar():
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        d = DATA[seed]
        models = sorted(d)
        ps, dis, sig = [], [], []
        for x, y in itertools.combinations(models, 2):
            cx = d[x][0] == d[x][2]
            cy = d[y][0] == d[y][2]
            b_ = int((cx & ~cy).sum())
            c_ = int((~cx & cy).sum())
            if b_ + c_ == 0:
                p = 1.0
            elif b_ + c_ < 25:
                p = stats.binomtest(b_, b_ + c_, 0.5).pvalue
            else:
                p = float(stats.chi2.sf((abs(b_ - c_) - 1) ** 2 / (b_ + c_), 1))
            ps.append(max(p, 1e-12))
            dis.append(int((d[x][0] != d[y][0]).sum()))
            sig.append(p <= 0.05)
        ps, dis, sig = np.array(ps), np.array(dis), np.array(sig)

        ax.scatter(ps[sig], dis[sig], s=9, c=C_GREY, alpha=.55,
                   lw=0, label="distinguishable")
        ax.scatter(ps[~sig], dis[~sig], s=11, c=C_ORANGE, alpha=.85,
                   lw=0, label="$p>0.05$")
        ax.axvline(0.05, color="k", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("McNemar $p$")
        ax.set_title(f"seed {seed[4:]}  ({int((~sig).sum())}/153 not separable)")
        ax.grid(lw=0.4, color="#EEEEEE")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("images the pair predicts\ndifferently (of 3552)")
    axes[0].legend(frameon=False, loc="lower left", handletextpad=.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR / "fig2_mcnemar.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig2_mcnemar.pdf")


# --------------------------------------------------------------- hình 3
def fig_diversity():
    """Phiên bản trung thực: độ bất đồng đo trên VALIDATION, lợi ích trên TEST.

    Bản trước đo cả hai trên test, nên quan hệ nhìn thấy có phần là do rò
    thông tin test. Bản này không có chuyện đó."""
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
    cmap = {1: C_GREY, 2: C_BLUE, 3: C_ORANGE}

    for ax, seed in zip(axes, SEEDS):
        d = DATA[seed]
        v = load_val(seed)
        models = sorted(set(d) & set(v))
        y_true = d[models[0]][2]
        yv = v[models[0]][2]
        acc = {m: float((d[m][0] == y_true).mean() * 100) for m in models}
        vacc = {m: float((v[m][0] == yv).mean() * 100) for m in models}
        dis_pair = {frozenset((x, y)): float((v[x][0] != v[y][0]).mean())
                    for x, y in itertools.combinations(models, 2)}

        X, Y, Cc = [], [], []
        for combo in itertools.combinations(models, 3):
            p = sum(d[m][1] for m in combo) / 3
            # mốc so sánh là model mà VAL cho là tốt nhất trong bộ ba
            base = acc[max(combo, key=lambda m: vacc[m])]
            gain = float((p.argmax(1) == y_true).mean() * 100) - base
            mdis = np.mean([dis_pair[frozenset(c)]
                            for c in itertools.combinations(combo, 2)])
            X.append(mdis); Y.append(gain)
            Cc.append(len({backbone_of(m) for m in combo}))
        X, Y, Cc = np.array(X), np.array(Y), np.array(Cc)

        for nb in (1, 2, 3):
            s = Cc == nb
            ax.scatter(X[s], Y[s], s=7, c=cmap[nb], alpha=.6, lw=0,
                       label=f"{nb} backbone" + ("s" if nb > 1 else ""))

        b1, b0 = np.polyfit(X, Y, 1)
        xx = np.linspace(X.min(), X.max(), 50)
        ax.plot(xx, b0 + b1 * xx, color="k", lw=1.0, ls="--")
        ax.axhline(0, color="k", lw=0.6)
        r = stats.spearmanr(X, Y)
        ax.set_title(f"seed {seed[4:]}   $\\rho$={r.statistic:+.2f}")
        ax.set_xlabel("mean pairwise disagreement (validation)")
        ax.grid(lw=0.4, color="#EEEEEE")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("test gain over\nbest member (points)")
    axes[1].legend(frameon=False, loc="lower right", handletextpad=.3,
                   borderpad=.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR / "fig3_diversity.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig3_diversity.pdf")


# --------------------------------------------------------------- hình 4
def fig_size():
    """Cỡ ensemble: chọn bằng val, chấm trên test."""
    df = pd.read_csv(HERE / "honest_sizes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        s = df[df.seed == seed].sort_values("k")
        ax.plot(s.k, s.per_backbone, "o-", color=C_ORANGE, lw=1.2, ms=4,
                label="one per backbone")
        ax.plot(s.k, s.diverse, "s-", color=C_BLUE, lw=1.2, ms=4,
                label="most diverse, $\\leq$2 pts")
        ax.plot(s.k, s.random, "^--", color=C_GREY, lw=1.0, ms=4,
                label="random $k$")
        ax.set_xticks(range(2, 7))
        ax.set_xlabel("ensemble size $k$")
        ax.set_title(f"seed {seed[4:]}")
        ax.grid(lw=0.4, color="#EEEEEE")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("test accuracy (\\%)")
    axes[0].legend(frameon=False, loc="lower right", handletextpad=.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR / "fig4_size.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig4_size.pdf")


if __name__ == "__main__":
    fig_head_gap()
    fig_mcnemar()
    fig_diversity()
    fig_size()
    print(f"-> {FIGDIR}")
