"""
Giao thức trung thực: chọn thành viên ensemble CHỈ bằng tập validation,
rồi mới chấm trên tập test.

Phân tích ở analysis_02 đo độ đa dạng trên chính tập test, nên quy tắc chọn
rút ra từ đó có rò thông tin test. Script này làm lại cho sạch, và trả lời
hai câu:

  1. Độ bất đồng đo trên VAL có còn dự báo được lợi ích ensemble trên TEST?
     (nếu không thì luận điểm của bài chỉ là hiện tượng của riêng tập test)
  2. Các quy tắc chọn, khi chỉ được nhìn VAL, còn hiệu quả không?

Thêm phần quét cỡ ensemble k = 2..6 để bỏ ràng buộc "cố định 3 model".
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
HEADS = ["SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"]
RNG = np.random.default_rng(36)


def backbone_of(m):
    for h in HEADS:
        if m.endswith("_" + h):
            return m[: -len(h) - 1]
    return m


def load(folder):
    out = {}
    for p in sorted(folder.glob("preds_*.npz")):
        d = np.load(p)
        out[p.stem.replace("preds_", "")] = (
            d["y_pred"].astype(int), d["y_prob"].astype(np.float64),
            d["y_true"].astype(int))
    return out


def soft_vote(d, combo):
    return (sum(d[m][1] for m in combo) / len(combo)).argmax(1)


def run_seed(seed, out_lines):
    val = load(HERE / f"results_val_{seed}")
    test = load(HERE / f"results_online_{seed}")
    models = sorted(set(val) & set(test))
    assert len(models) == 18, f"{seed}: chỉ có {len(models)} model chung"

    yv = val[models[0]][2]
    yt = test[models[0]][2]
    vacc = {m: float((val[m][0] == yv).mean() * 100) for m in models}
    tacc = {m: float((test[m][0] == yt).mean() * 100) for m in models}

    vdis = {frozenset((a, b)): float((val[a][0] != val[b][0]).mean())
            for a, b in itertools.combinations(models, 2)}

    out_lines.append("")
    out_lines.append("=" * 74)
    out_lines.append(f"{seed}   n_val={len(yv)}  n_test={len(yt)}")
    out_lines.append("=" * 74)

    r = stats.pearsonr([vacc[m] for m in models], [tacc[m] for m in models])
    out_lines.append(f"Tương quan accuracy VAL vs TEST trên 18 model: "
                     f"r={r.statistic:.3f} (p={r.pvalue:.2g})")

    # ---------- câu 1: bất đồng đo trên VAL có dự báo lợi ích trên TEST?
    rows = []
    for combo in itertools.combinations(models, 3):
        ens_t = float((soft_vote(test, combo) == yt).mean() * 100)
        rows.append({
            "combo": " + ".join(combo),
            "val_mean_acc": float(np.mean([vacc[m] for m in combo])),
            "val_dis": float(np.mean([vdis[frozenset(c)]
                                      for c in itertools.combinations(combo, 2)])),
            "test_ens": ens_t,
            "test_gain_vs_valbest": ens_t - tacc[max(combo, key=lambda m: vacc[m])],
            "nb": len({backbone_of(m) for m in combo}),
        })
    T = pd.DataFrame(rows)

    y = T.test_gain_vs_valbest.values
    def fit(cols):
        X = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        dof = len(y) - X.shape[1]
        se = np.sqrt(np.diag(res @ res / dof * np.linalg.inv(X.T @ X)))
        t = beta / se
        p = 2 * stats.t.sf(np.abs(t), dof)
        r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
        return beta, se, p, r2

    out_lines.append("")
    out_lines.append("Dự báo LỢI ÍCH TRÊN TEST bằng các biến đo TRÊN VAL "
                     "(816 tổ hợp):")
    for name, cols in (
        ("accuracy TB (val)", [T.val_mean_acc.values]),
        ("bất đồng (val)", [T.val_dis.values]),
        ("cả hai", [T.val_mean_acc.values, T.val_dis.values]),
    ):
        beta, se, p, r2 = fit(cols)
        parts = ", ".join(f"{b:+.3f} (p={pp:.2g})" for b, pp in zip(beta[1:], p[1:]))
        out_lines.append(f"  {name:<20} R2={r2:.3f}   hệ số {parts}")

    rho = stats.spearmanr(T.val_dis, T.test_gain_vs_valbest)
    out_lines.append(f"  Spearman bất đồng(val) vs lợi ích(test): "
                     f"rho={rho.statistic:+.3f}, p={rho.pvalue:.2g}")

    # ---------- câu 2: quy tắc chọn chỉ nhìn VAL
    by_vacc = sorted(models, key=lambda m: -vacc[m])
    val_best = by_vacc[0]

    def pick_per_backbone(k):
        seen, out = set(), []
        for m in by_vacc:
            b = backbone_of(m)
            if b not in seen:
                seen.add(b); out.append(m)
            if len(out) == k:
                break
        return tuple(out)

    def pick_diverse_among_good(k, margin=2.0):
        pool = [m for m in models if vacc[m] >= vacc[val_best] - margin]
        if len(pool) < k:
            pool = by_vacc[:k]
        best, bestd = None, -1
        for combo in itertools.combinations(pool, k):
            dd = np.mean([vdis[frozenset(c)]
                          for c in itertools.combinations(combo, 2)])
            if dd > bestd:
                best, bestd = combo, dd
        return best

    def pick_most_diverse(k):
        best, bestd = None, -1
        for combo in itertools.combinations(models, k):
            dd = np.mean([vdis[frozenset(c)]
                          for c in itertools.combinations(combo, 2)])
            if dd > bestd:
                best, bestd = combo, dd
        return best

    rules = [
        ("Top-3 by val accuracy", tuple(by_vacc[:3])),
        ("One per backbone, val accuracy", pick_per_backbone(3)),
        ("Most val-diverse overall", pick_most_diverse(3)),
        ("Most val-diverse within 2 pts", pick_diverse_among_good(3)),
    ]

    out_lines.append("")
    out_lines.append(f"Model đơn tốt nhất theo VAL: {val_best} "
                     f"(val {vacc[val_best]:.2f}%, test {tacc[val_best]:.2f}%)")
    out_lines.append("")
    out_lines.append(f"  {'quy tắc (chỉ nhìn val)':<34}{'test':>7}{'gain':>7}  thành viên")
    sel_rows = []
    for name, combo in rules:
        a = float((soft_vote(test, combo) == yt).mean() * 100)
        g = a - tacc[val_best]
        out_lines.append(f"  {name:<34}{a:7.2f}{g:+7.2f}  "
                         + ", ".join(sorted(backbone_of(m) for m in combo)))
        sel_rows.append({"seed": seed, "rule": name, "test_acc": a, "gain": g})

    rnd = [float((soft_vote(test, tuple(RNG.choice(models, 3, replace=False))) == yt).mean() * 100)
           for _ in range(200)]
    out_lines.append(f"  {'Random triple (mean of 200)':<34}{np.mean(rnd):7.2f}"
                     f"{np.mean(rnd)-tacc[val_best]:+7.2f}")
    sel_rows.append({"seed": seed, "rule": "Random triple (mean of 200)",
                     "test_acc": float(np.mean(rnd)),
                     "gain": float(np.mean(rnd) - tacc[val_best])})

    orc = T.sort_values("test_ens", ascending=False).iloc[0]
    out_lines.append(f"  {'Oracle on test (not usable)':<34}{orc.test_ens:7.2f}"
                     f"{orc.test_ens - tacc[val_best]:+7.2f}")
    sel_rows.append({"seed": seed, "rule": "Oracle on test (not usable)",
                     "test_acc": float(orc.test_ens),
                     "gain": float(orc.test_ens - tacc[val_best])})

    # ---------- quét cỡ ensemble
    out_lines.append("")
    out_lines.append("  Cỡ ensemble (chọn bằng val, chấm trên test):")
    out_lines.append(f"    {'k':>2}{'per-backbone':>14}{'diverse<=2pt':>14}{'random':>9}")
    size_rows = []
    for k in range(2, 7):
        a1 = float((soft_vote(test, pick_per_backbone(k)) == yt).mean() * 100)
        a2 = float((soft_vote(test, pick_diverse_among_good(k)) == yt).mean() * 100)
        rr = [float((soft_vote(test, tuple(RNG.choice(models, k, replace=False))) == yt).mean() * 100)
              for _ in range(150)]
        out_lines.append(f"    {k:>2}{a1:14.2f}{a2:14.2f}{np.mean(rr):9.2f}")
        size_rows.append({"seed": seed, "k": k, "per_backbone": a1,
                          "diverse": a2, "random": float(np.mean(rr))})

    return T, pd.DataFrame(sel_rows), pd.DataFrame(size_rows), tacc[val_best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["seed36", "seed42"])
    a = ap.parse_args()

    lines, sels, sizes = [], [], []
    for s in a.seeds:
        T, sel, size, _ = run_seed(s, lines)
        T.to_csv(HERE / f"honest_triples_{s}.csv", index=False)
        sels.append(sel); sizes.append(size)

    pd.concat(sels).to_csv(HERE / "honest_selection.csv", index=False)
    pd.concat(sizes).to_csv(HERE / "honest_sizes.csv", index=False)

    text = "\n".join(lines)
    (HERE / "honest_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
