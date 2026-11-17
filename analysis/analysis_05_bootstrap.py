"""
Độ nhiễu của việc CHỌN thành viên khi chỉ có một tập validation 3.511 ảnh.

Bài (mục Limitations) nói: nhiễu khi ước lượng độ đa dạng trên tập val cỡ này
đủ để làm cùn quy tắc chọn kiểu argmax. Script này đo thẳng điều đó: lấy mẫu
lại (bootstrap) tập val B lần, mỗi lần chạy lại từng quy tắc chọn, rồi chấm
tổ hợp được chọn trên test (test cố định, không lấy mẫu lại).

Báo cáo cho mỗi quy tắc:
  - tần suất tổ hợp hay được chọn nhất (độ ổn định của lựa chọn),
  - số tổ hợp khác nhau được chọn,
  - trung bình / độ lệch chuẩn / khoảng 5-95% của accuracy test.
"""

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_03_honest import backbone_of, load

HERE = Path(__file__).parent
from paths import OUT, test_dir, val_dir


def run_seed(seed, B, lines):
    rng = np.random.default_rng(36)
    val = load(val_dir(seed))
    test = load(test_dir(seed))
    models = sorted(set(val) & set(test))
    assert len(models) == 18
    M = len(models)
    yv, yt = val[models[0]][2], test[models[0]][2]
    PV = np.stack([val[m][0] for m in models])          # (M, Nv) nhãn dự đoán
    CV = (PV == yv)                                      # đúng/sai trên val
    PT = np.stack([test[m][1] for m in models])          # (M, Nt, C)
    triples = list(itertools.combinations(range(M), 3))
    pairs_of = [list(itertools.combinations(t, 2)) for t in triples]
    bb = [backbone_of(m) for m in models]

    cache = {}
    def test_acc(combo):
        combo = tuple(sorted(combo))
        if combo not in cache:
            cache[combo] = float((PT[list(combo)].mean(0).argmax(1) == yt).mean() * 100)
        return cache[combo]

    def rules(idx):
        acc = CV[:, idx].mean(1) * 100
        P = PV[:, idx]
        dis = np.zeros((M, M))
        for a, b in itertools.combinations(range(M), 2):
            dis[a, b] = dis[b, a] = (P[a] != P[b]).mean()
        tdis = np.array([np.mean([dis[a, b] for a, b in pp]) for pp in pairs_of])
        order = np.argsort(-acc, kind="stable")
        best = order[0]
        seen, pbb = set(), []
        for m in order:
            if bb[m] not in seen:
                seen.add(bb[m]); pbb.append(m)
            if len(pbb) == 3:
                break
        good = acc >= acc[best] - 2.0
        ok = np.array([all(good[list(t)]) for t in triples])
        if ok.sum() == 0:
            within = tuple(order[:3])
        else:
            within = triples[int(np.argmax(np.where(ok, tdis, -1)))]
        return {
            "Top-3 by val accuracy": tuple(order[:3]),
            "One per backbone, val accuracy": tuple(pbb),
            "Most val-diverse overall": triples[int(np.argmax(tdis))],
            "Most val-diverse within 2 pts": tuple(within),
            "_best_single": (best,),
        }

    full = rules(np.arange(len(yv)))
    picks = {k: [] for k in full}
    for _ in range(B):
        idx = rng.integers(0, len(yv), len(yv))
        for k, v in rules(idx).items():
            picks[k].append(tuple(sorted(v)))

    lines += ["", "=" * 86, f"{seed}  B={B} bootstrap của tập val (n={len(yv)})", "=" * 86]
    lines.append(f"  {'quy tắc':<34}{'gốc':>7}{'TB':>7}{'SD':>6}{'P5':>7}{'P95':>7}"
                 f"{'#tổ hợp':>9}{'mode %':>8}")
    rows = []
    for k, lst in picks.items():
        accs = np.array([test_acc(c) if len(c) == 3 else
                         float((PT[c[0]].argmax(1) == yt).mean() * 100) for c in lst])
        c = Counter(lst)
        mode_share = c.most_common(1)[0][1] / B * 100
        f = full[k]
        orig = test_acc(f) if len(f) == 3 else float((PT[f[0]].argmax(1) == yt).mean() * 100)
        name = "Best single model" if k == "_best_single" else k
        lines.append(f"  {name:<34}{orig:7.2f}{accs.mean():7.2f}{accs.std():6.2f}"
                     f"{np.percentile(accs, 5):7.2f}{np.percentile(accs, 95):7.2f}"
                     f"{len(c):>9}{mode_share:8.0f}")
        rows.append({"seed": seed, "rule": name, "full_val": orig, "mean": accs.mean(),
                     "sd": accs.std(), "p5": np.percentile(accs, 5),
                     "p95": np.percentile(accs, 95), "n_distinct": len(c),
                     "mode_share": mode_share})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["seed36", "seed42"])
    ap.add_argument("--B", type=int, default=500)
    a = ap.parse_args()
    lines = []
    df = pd.concat([run_seed(s, a.B, lines) for s in a.seeds])
    df.to_csv(OUT / "bootstrap_selection.csv", index=False)
    (OUT / "bootstrap_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
