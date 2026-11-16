"""Kiểm tra tính nhất quán của dự đoán tải từ Kaggle.

Hai câu hỏi phải trả lời TRƯỚC khi phân tích:
  1. 46 file có dùng chung một thứ tự ảnh không? Nếu không thì mọi phép so
     từng cặp đều vô nghĩa.
  2. Lần chạy nào ứng với Bảng 1 của bài?
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent / "results"

# Bảng 1 trong bài (accuracy test, %).
TABLE1 = {
    "efficient_net_SIMPLE_FC": 90.51,
    "efficient_net_GAP_BN_FC_Drop": 92.03,
    "efficient_net_DUAL_POOLING_FC": 92.12,
    "dense_net_SIMPLE_FC": 91.08,
    "dense_net_GAP_BN_FC_Drop": 92.20,
    "dense_net_DUAL_POOLING_FC": 91.89,
    "xception_SIMPLE_FC": 90.32,
    "xception_GAP_BN_FC_Drop": 91.10,
    "xception_DUAL_POOLING_FC": 90.60,
    "resnet_50_SIMPLE_FC": 89.39,
    "resnet_50_GAP_BN_FC_Drop": 90.77,
    "resnet_50_DUAL_POOLING_FC": 90.57,
    "inception_v3_SIMPLE_FC": 88.40,
    "inception_v3_GAP_BN_FC_Drop": 90.03,
    "inception_v3_DUAL_POOLING_FC": 88.82,
    "mobilenet_v3_SIMPLE_FC": 84.91,
    "mobilenet_v3_GAP_BN_FC_Drop": 87.95,
    "mobilenet_v3_DUAL_POOLING_FC": 87.75,
}


def main():
    files = sorted(ROOT.rglob("preds_*.npz"))
    if not files:
        sys.exit(f"Không có file nào trong {ROOT}")

    data = {}
    for f in files:
        seed = f.parent.name
        model = f.stem.replace("preds_", "")
        d = np.load(f)
        data[(seed, model)] = (d["y_true"].astype(int), d["y_pred"].astype(int))

    # ---------------------------------------------------- 1. thứ tự ảnh
    print("=" * 70)
    print("1. THỨ TỰ ẢNH — y_true có giống nhau giữa các file không?")
    print("=" * 70)

    groups = defaultdict(list)
    for (seed, model), (y_true, _) in data.items():
        # chữ ký của một thứ tự: băm toàn bộ vector nhãn
        sig = hash(y_true.tobytes())
        groups[sig].append(f"{seed}/{model}")

    print(f"Số thứ tự khác nhau tìm thấy: {len(groups)}")
    for i, (sig, members) in enumerate(groups.items(), 1):
        seeds = sorted({m.split('/')[0] for m in members})
        print(f"\n  Nhóm {i}: {len(members)} file — seed {', '.join(seeds)}")
        if len(groups) > 1:
            for m in sorted(members)[:6]:
                print(f"      {m}")
            if len(members) > 6:
                print(f"      ... và {len(members)-6} file nữa")

    if len(groups) == 1:
        print("\n  => Tất cả dùng chung một thứ tự. So từng cặp được.")
    else:
        print("\n  => CẢNH BÁO: có nhiều thứ tự khác nhau.")
        print("     Chỉ được so các file trong CÙNG một nhóm.")

    # phân bố nhãn có giống nhau không (nếu khác thứ tự nhưng cùng tập)
    if len(groups) > 1:
        print("\n  Phân bố nhãn theo nhóm (để biết có cùng tập test không):")
        for i, (sig, members) in enumerate(groups.items(), 1):
            seed, model = members[0].split("/")
            y = data[(seed, model)][0]
            cnt = np.bincount(y, minlength=36)
            print(f"    Nhóm {i}: n={len(y)}, tổng theo lớp (10 lớp đầu) {cnt[:10].tolist()}")

    # ---------------------------------------------- 2. khớp Bảng 1 chưa
    print()
    print("=" * 70)
    print("2. SO VỚI BẢNG 1 CỦA BÀI")
    print("=" * 70)

    by_seed = defaultdict(dict)
    for (seed, model), (y_true, y_pred) in data.items():
        by_seed[seed][model] = float((y_pred == y_true).mean() * 100)

    for seed in sorted(by_seed):
        accs = by_seed[seed]
        common = [m for m in TABLE1 if m in accs]
        if not common:
            continue
        deltas = [accs[m] - TABLE1[m] for m in common]
        print(f"\n{seed}  ({len(common)}/18 cấu hình so được)")
        print(f"  lệch trung bình {np.mean(deltas):+.2f} điểm, "
              f"khoảng [{min(deltas):+.2f}, {max(deltas):+.2f}]")
        n_match = sum(1 for d in deltas if abs(d) <= 0.1)
        print(f"  khớp trong 0,1 điểm: {n_match}/{len(common)}")
        if n_match == len(common):
            print("  => ĐÂY LÀ LẦN CHẠY CỦA BẢNG 1")
        elif np.mean(deltas) < -0.5:
            print("  => Thấp hơn Bảng 1 một cách hệ thống: khác lần chạy "
                  "(nhiều khả năng online-augment, còn bài dùng offline).")

    print("\nChi tiết từng cấu hình (seed36):")
    print(f"  {'cấu hình':<34} {'tải về':>8} {'Bảng 1':>8} {'lệch':>8}")
    for m in TABLE1:
        if m in by_seed.get("seed36", {}):
            a, t = by_seed["seed36"][m], TABLE1[m]
            print(f"  {m:<34} {a:8.2f} {t:8.2f} {a-t:+8.2f}")


if __name__ == "__main__":
    main()
