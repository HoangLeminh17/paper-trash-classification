"""
Dựng bộ 18 cấu hình đầy đủ từ các lần chạy online-augment (tháng 6).

Lần chạy offline (nguồn của Bảng 1) chỉ lưu dự đoán cho 6/18 cấu hình, không
đủ để phân tích tổ hợp — với 6 model chỉ có C(6,3)=20 bộ ba thay vì 816.
Lần chạy online thì đủ 18 cấu hình cho seed36 và seed42.

Script gom chúng theo seed, và kiểm tra: cấu hình nào xuất hiện ở hai notebook
thì dự đoán phải trùng khít từng ảnh, nếu không thì đó là hai lần train khác
nhau và phải dừng lại xem xét.
"""

import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "results_by_notebook"

ONLINE_KERNELS = [
    "36-labels-gpu-with-augment-8-6-2026",
    "36labe-gpu-with-augment-9-6-2026-from-seed36-dense",
    "36labe-gpu-with-augment9-6-2026-from-seed42-resnet",
]

BACKBONES = ["efficient_net", "mobilenet_v3", "resnet_50",
             "dense_net", "inception_v3", "xception"]
HEADS = ["SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"]
WANTED = {f"{b}_{h}" for b in BACKBONES for h in HEADS}


def main():
    # (seed, model) -> [(slug, path), ...]
    found = defaultdict(list)
    for slug in ONLINE_KERNELS:
        d = RAW / slug
        if not d.exists():
            print(f"[bỏ qua] chưa tải {slug}")
            continue
        for p in d.glob("preds_seed*_*.npz"):
            stem = p.stem[len("preds_"):]
            seed, model = stem.split("_", 1)
            found[(seed, model)].append((slug, p))

    conflicts = []
    by_seed = defaultdict(dict)

    for (seed, model), entries in sorted(found.items()):
        if len(entries) > 1:
            # cùng cấu hình ở nhiều notebook: phải trùng khít từng ảnh
            arrs = [np.load(p)["y_pred"] for _, p in entries]
            if not all(np.array_equal(arrs[0], a) for a in arrs[1:]):
                accs = []
                for slug, p in entries:
                    d = np.load(p)
                    accs.append(
                        f"{slug}={float((d['y_pred']==d['y_true']).mean()*100):.2f}%"
                    )
                conflicts.append(f"{seed}/{model}: " + ", ".join(accs))
                continue
        by_seed[seed][model] = entries[0][1]

    if conflicts:
        print("XUNG ĐỘT — cùng cấu hình nhưng dự đoán khác nhau:\n")
        for c in conflicts:
            print("  " + c)
        print("\nĐây là hai lần train khác nhau, không phải bản sao. Phải chọn tay.\n")

    for seed in sorted(by_seed):
        models = by_seed[seed]
        missing = WANTED - set(models)
        out = HERE / f"results_online_{seed}"
        status = "ĐỦ 18/18" if not missing else f"{len(models)}/18, thiếu {len(missing)}"
        print(f"{seed}: {status}")
        if missing:
            print(f"   thiếu: {', '.join(sorted(missing))}")
            continue
        out.mkdir(exist_ok=True)
        for model, p in models.items():
            shutil.copy(p, out / f"preds_{model}.npz")
        print(f"   -> {out.name}")

    print("\nChạy tiếp:")
    for seed in sorted(by_seed):
        if not (WANTED - set(by_seed[seed])):
            print(f"    python analysis_02_diversity.py --dir results_online_{seed}")


if __name__ == "__main__":
    main()
