"""
Gộp CÓ TRỌNG SỐ, học trọng số chỉ trên validation, chấm trên test.

Bài (mục Limitations) nói: dùng soft voting không trọng số; trọng số học được
có thể nâng số tuyệt đối và có thể đổi quy tắc chọn nào thắng. Script này kiểm
tra đúng câu đó.

Trọng số: w = softmax(theta), tối ưu log-likelihood trên VAL (trơn, không
overfit như tối ưu thẳng accuracy). So sánh với soft vote đều cho:
  - thành viên do từng quy tắc chọn (k = 3),
  - cỡ k = 2..6 theo hai quy tắc chính,
  - toàn bộ 18 model (trọng số học được có thay được việc chọn thành viên?).
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

from analysis_03_honest import HEADS, backbone_of, load

HERE = Path(__file__).parent
from paths import OUT, test_dir, val_dir
EPS = 1e-12


def fit_weights(P_val, y_val):
    """P_val: (k, N, C). Trả về trọng số >= 0, tổng 1, tối ưu NLL trên val."""
    k = P_val.shape[0]
    if k == 1:
        return np.ones(1)
    py = P_val[:, np.arange(len(y_val)), y_val]  # (k, N): xác suất lớp đúng

    def nll(theta):
        w = softmax(theta)
        q = w @ py
        return -np.mean(np.log(q + EPS))

    def grad(theta):
        w = softmax(theta)
        q = w @ py + EPS
        g_w = -(py / q).mean(1)               # dNLL/dw
        return w * (g_w - w @ g_w)            # qua softmax

    r = minimize(nll, np.zeros(k), jac=grad, method="L-BFGS-B")
    return softmax(r.x)


def acc_of(P, w, y):
    return float(((np.tensordot(w, P, 1)).argmax(1) == y).mean() * 100)


def run_seed(seed, lines):
    val = load(val_dir(seed))
    test = load(test_dir(seed))
    models = sorted(set(val) & set(test))
    assert len(models) == 18, f"{seed}: {len(models)} model"
    yv, yt = val[models[0]][2], test[models[0]][2]
    PV = {m: val[m][1] for m in models}
    PT = {m: test[m][1] for m in models}
    vacc = {m: float((val[m][0] == yv).mean() * 100) for m in models}
    tacc = {m: float((test[m][0] == yt).mean() * 100) for m in models}
    vdis = {frozenset((a, b)): float((val[a][0] != val[b][0]).mean())
            for a, b in itertools.combinations(models, 2)}
    by_vacc = sorted(models, key=lambda m: -vacc[m])
    best = by_vacc[0]

    def mdis(c):
        return np.mean([vdis[frozenset(p)] for p in itertools.combinations(c, 2)])

    def per_backbone(k):
        seen, out = set(), []
        for m in by_vacc:
            if backbone_of(m) not in seen:
                seen.add(backbone_of(m)); out.append(m)
            if len(out) == k:
                break
        return tuple(out)

    def diverse_within(k, margin=2.0):
        pool = [m for m in models if vacc[m] >= vacc[best] - margin]
        if len(pool) < k:
            pool = by_vacc[:k]
        return max(itertools.combinations(pool, k), key=mdis)

    def most_diverse(k):
        return max(itertools.combinations(models, k), key=mdis)

    def evaluate(combo):
        Pv = np.stack([PV[m] for m in combo]); Pt = np.stack([PT[m] for m in combo])
        k = len(combo)
        w = fit_weights(Pv, yv)
        return acc_of(Pt, np.full(k, 1 / k), yt), acc_of(Pt, w, yt), w

    lines += ["", "=" * 78, f"{seed}  (model đơn tốt nhất theo val: {best}, test {tacc[best]:.2f})", "=" * 78]
    rows = []

    lines.append(f"  {'quy tắc, k=3':<34}{'đều':>8}{'trọng số':>10}{'chênh':>8}  trọng số")
    for name, combo in (
        ("Top-3 by val accuracy", tuple(by_vacc[:3])),
        ("One per backbone, val accuracy", per_backbone(3)),
        ("Most val-diverse overall", most_diverse(3)),
        ("Most val-diverse within 2 pts", diverse_within(3)),
    ):
        u, wv, w = evaluate(combo)
        lines.append(f"  {name:<34}{u:8.2f}{wv:10.2f}{wv-u:+8.2f}  "
                     + ", ".join(f"{backbone_of(m)}:{x:.2f}" for m, x in zip(combo, w)))
        rows.append({"seed": seed, "setting": name, "k": 3, "unweighted": u,
                     "weighted": wv, "best_single": tacc[best]})

    lines.append("")
    lines.append(f"  {'k':>3}{'per-bb đều':>12}{'per-bb w':>10}{'div đều':>10}{'div w':>9}")
    for k in range(2, 7):
        u1, w1, _ = evaluate(per_backbone(k))
        u2, w2, _ = evaluate(diverse_within(k))
        lines.append(f"  {k:>3}{u1:12.2f}{w1:10.2f}{u2:10.2f}{w2:9.2f}")
        rows.append({"seed": seed, "setting": "One per backbone, val accuracy", "k": k,
                     "unweighted": u1, "weighted": w1, "best_single": tacc[best]})
        rows.append({"seed": seed, "setting": "Most val-diverse within 2 pts", "k": k,
                     "unweighted": u2, "weighted": w2, "best_single": tacc[best]})

    u, wv, w = evaluate(tuple(models))
    top = sorted(zip(models, w), key=lambda t: -t[1])
    n_eff = 1 / np.sum(w ** 2)
    lines.append("")
    lines.append(f"  Cả 18 model: đều {u:.2f}, trọng số {wv:.2f} "
                 f"(số model hiệu dụng 1/sum(w^2) = {n_eff:.1f})")
    lines.append("    trọng số lớn nhất: " + ", ".join(f"{m}:{x:.2f}" for m, x in top[:6]))
    rows.append({"seed": seed, "setting": "All 18 models", "k": 18, "unweighted": u,
                 "weighted": wv, "best_single": tacc[best]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["seed36", "seed42"])
    a = ap.parse_args()
    lines = []
    df = pd.concat([run_seed(s, lines) for s in a.seeds])
    df.to_csv(OUT / "weighted_combination.csv", index=False)
    (OUT / "weighted_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
