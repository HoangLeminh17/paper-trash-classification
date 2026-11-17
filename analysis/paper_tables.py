"""Sinh đúng các con số sẽ đưa vào bài IDA, từ hai seed đầy đủ."""

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
from paths import OUT, test_dir, val_dir
SEEDS = ["seed36", "seed42"]
BACKBONES = ["efficient_net", "dense_net", "xception", "resnet_50",
             "inception_v3", "mobilenet_v3"]
HEADS = ["SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"]


def load(seed):
    out = {}
    for p in (test_dir(seed)).glob("preds_*.npz"):
        d = np.load(p)
        out[p.stem.replace("preds_", "")] = (
            d["y_pred"].astype(int), d["y_prob"].astype(np.float64),
            d["y_true"].astype(int),
        )
    return out


def main():
    data = {s: load(s) for s in SEEDS}
    train = defaultdict(dict)
    with open(OUT / "train_acc.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            train[r["seed"]][r["model"]] = float(r["train_acc"])

    print("=" * 76)
    print("BẢNG 1 — accuracy test theo cấu hình, hai seed, kèm độ chênh train-test")
    print("=" * 76)
    print(f"{'backbone':<15}{'head':<17}{'s36':>7}{'s42':>7}{'TB':>7}"
          f"{'train s36':>11}{'gap s36':>9}")
    rows = []
    for b in BACKBONES:
        for h in HEADS:
            m = f"{b}_{h}"
            a36 = (data["seed36"][m][0] == data["seed36"][m][2]).mean() * 100
            a42 = (data["seed42"][m][0] == data["seed42"][m][2]).mean() * 100
            tr = train["seed36"].get(m, float("nan"))
            print(f"{b:<15}{h:<17}{a36:7.2f}{a42:7.2f}{(a36+a42)/2:7.2f}"
                  f"{tr:11.2f}{tr-a36:9.2f}")
            rows.append({"backbone": b, "head": h, "s36": a36, "s42": a42,
                         "train36": tr, "gap36": tr - a36})
    df = pd.DataFrame(rows)

    print("\n" + "=" * 76)
    print("HIỆU ỨNG CỦA HEAD")
    print("=" * 76)
    g = df.groupby("head", sort=False).agg(acc_s36=("s36", "mean"), acc_s42=("s42", "mean"),
                               gap=("gap36", "mean"),
                               gap_min=("gap36", "min"), gap_max=("gap36", "max"))
    print(g.reindex(HEADS).round(2).to_string())

    def cell(b, h, col):
        # df["head"] chứ không phải df.head: .head là phương thức của DataFrame
        sel = df[(df["backbone"] == b) & (df["head"] == h)]
        return float(sel[col].iloc[0])

    print("\n  GAP_BN_FC_Drop so với SIMPLE_FC, từng backbone (điểm):")
    for b in BACKBONES:
        d36 = cell(b, "GAP_BN_FC_Drop", "s36") - cell(b, "SIMPLE_FC", "s36")
        d42 = cell(b, "GAP_BN_FC_Drop", "s42") - cell(b, "SIMPLE_FC", "s42")
        print(f"    {b:<15} s36 {d36:+6.2f}   s42 {d42:+6.2f}")

    # Wilcoxon ghép cặp trên 6 backbone x 2 seed
    pairs = []
    for col in ("s36", "s42"):
        for b in BACKBONES:
            pairs.append((cell(b, "GAP_BN_FC_Drop", col),
                          cell(b, "SIMPLE_FC", col)))
    a, bb = zip(*pairs)
    w = stats.wilcoxon(a, bb)
    print(f"\n  Wilcoxon ghép cặp (12 cặp backbone x seed): "
          f"W={w.statistic:.0f}, p={w.pvalue:.4g}")

    print("\n" + "=" * 76)
    print("McNEMAR — cặp không phân biệt được vẫn sai ở đâu")
    print("=" * 76)
    for s in SEEDS:
        d = data[s]
        models = sorted(d)
        rows2 = []
        for x, y in itertools.combinations(models, 2):
            cx = d[x][0] == d[x][2]
            cy = d[y][0] == d[y][2]
            b_ = int((cx & ~cy).sum()); c_ = int((~cx & cy).sum())
            if b_ + c_ == 0:
                p = 1.0
            elif b_ + c_ < 25:
                p = stats.binomtest(b_, b_ + c_, 0.5).pvalue
            else:
                p = float(stats.chi2.sf((abs(b_ - c_) - 1) ** 2 / (b_ + c_), 1))
            rows2.append({"p": p, "dis": int((d[x][0] != d[y][0]).sum()),
                          "df": int((~cx & ~cy).sum())})
        r2 = pd.DataFrame(rows2)
        ns = r2[r2.p > 0.05]
        print(f"  {s}: {len(ns)}/{len(r2)} cặp p>0,05; "
              f"trong đó số ảnh đoán khác nhau TB {ns.dis.mean():.0f} "
              f"({ns.dis.min()}–{ns.dis.max()}), "
              f"cùng sai TB {ns['df'].mean():.0f}")

    print("\n" + "=" * 76)
    print("ENSEMBLE — trần trên và các cách chọn")
    print("=" * 76)
    for s in SEEDS:
        d = data[s]
        models = sorted(d)
        y_true = d[models[0]][2]
        acc = {m: float((d[m][0] == y_true).mean() * 100) for m in models}
        best = max(acc, key=acc.get)
        def backbone_of(m):
            for h in HEADS:
                if m.endswith("_" + h):
                    return m[: -len(h) - 1]
            return m

        recs = []
        for combo in itertools.combinations(models, 3):
            p = sum(d[m][1] for m in combo) / 3
            recs.append({
                "combo": " + ".join(combo),
                "acc": float((p.argmax(1) == y_true).mean() * 100),
                "nb": len({backbone_of(m) for m in combo}),
            })
        t = pd.DataFrame(recs).sort_values("acc", ascending=False)
        print(f"\n  {s}: model đơn tốt nhất {best} = {acc[best]:.2f}%")
        print(f"       ensemble tốt nhất {t.iloc[0].acc:.2f}% "
              f"(+{t.iloc[0].acc - acc[best]:.2f})")
        print(f"       trung vị 816 tổ hợp {t.acc.median():.2f}%, "
              f"tệ nhất {t.acc.min():.2f}%")
        n_beat = int((t.acc > acc[best]).sum())
        print(f"       số tổ hợp vượt model đơn tốt nhất: {n_beat}/816 "
              f"({n_beat/816*100:.0f}%)")


if __name__ == "__main__":
    main()
