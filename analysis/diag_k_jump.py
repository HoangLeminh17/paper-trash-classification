"""Vì sao seed36 nhảy +0,79 từ k=3 lên k=4 còn seed42 chỉ +0,08?

Quy tắc "mỗi backbone một model" xếp backbone theo val accuracy của model tốt
nhất trong backbone đó, rồi lấy k backbone đầu. Nếu backbone thứ 3 và thứ 4
gần bằng nhau về val accuracy thì việc cái nào được chọn ở k=3 là ngẫu nhiên
theo seed — và nếu hai cái đó khác nhau nhiều về khả năng bổ trợ thì k=3 của
seed này tốt, của seed kia dở.

Script in ra: thứ hạng backbone, khoảng cách val giữa chúng, backbone được
thêm ở mỗi k, và accuracy test tương ứng.
"""

import itertools
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
HEADS = ["SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"]
SEEDS = ["seed36", "seed42"]


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


for seed in SEEDS:
    val = load(HERE / f"results_val_{seed}")
    test = load(HERE / f"results_online_{seed}")
    models = sorted(set(val) & set(test))
    yv, yt = val[models[0]][2], test[models[0]][2]
    vacc = {m: float((val[m][0] == yv).mean() * 100) for m in models}
    tacc = {m: float((test[m][0] == yt).mean() * 100) for m in models}

    # model tốt nhất (theo val) của từng backbone, rồi xếp hạng backbone
    best_of = {}
    for m in models:
        b = backbone_of(m)
        if b not in best_of or vacc[m] > vacc[best_of[b]]:
            best_of[b] = m
    order = sorted(best_of, key=lambda b: -vacc[best_of[b]])

    print("=" * 72)
    print(f"{seed}")
    print("=" * 72)
    print("Xếp hạng backbone theo val accuracy của model tốt nhất trong nó:")
    for i, b in enumerate(order, 1):
        m = best_of[b]
        head = m[len(b) + 1:]
        print(f"  {i}. {b:<15} {head:<17} val {vacc[m]:6.2f}  test {tacc[m]:6.2f}")

    d34 = vacc[best_of[order[2]]] - vacc[best_of[order[3]]]
    print(f"\n  Khoảng cách val giữa backbone hạng 3 và hạng 4: {d34:.2f} điểm")

    print("\n  Ensemble theo k (chọn bằng val, chấm trên test):")
    prev = None
    for k in range(2, 7):
        combo = tuple(best_of[b] for b in order[:k])
        p = sum(test[m][1] for m in combo) / k
        a = float((p.argmax(1) == yt).mean() * 100)
        added = order[k - 1] if k > 2 else f"{order[0]}+{order[1]}"
        delta = f"{a - prev:+.2f}" if prev is not None else "    "
        print(f"    k={k}  {a:6.2f}  {delta}   thêm: {added}")
        prev = a

    # bộ ba tốt nhất có thể có nếu đổi backbone hạng 3 bằng hạng 4
    alt = (best_of[order[0]], best_of[order[1]], best_of[order[3]])
    pa = sum(test[m][1] for m in alt) / 3
    aa = float((pa.argmax(1) == yt).mean() * 100)
    cur = (best_of[order[0]], best_of[order[1]], best_of[order[2]])
    pc = sum(test[m][1] for m in cur) / 3
    ac = float((pc.argmax(1) == yt).mean() * 100)
    print(f"\n  Nếu ở k=3 đổi backbone hạng 3 ({order[2]}) bằng hạng 4 ({order[3]}):")
    print(f"    hiện tại {ac:.2f}  ->  thay thế {aa:.2f}   ({aa - ac:+.2f})")
    print()
