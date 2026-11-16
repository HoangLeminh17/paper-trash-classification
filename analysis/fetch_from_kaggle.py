"""
Bước 0 — Tải dự đoán từ Kaggle, GIỮ NGUYÊN NGUỒN GỐC.

Bài học từ lần chạy đầu: nhiều notebook cùng sinh ra một cấu hình nhưng ở
những lần chạy khác nhau (online augment tháng 6 vs offline augment tháng 8),
accuracy lệch nhau tới 3 điểm. Gộp theo tên cấu hình rồi "lấy bản đầu tiên"
là trộn lẫn hai thí nghiệm — sai hoàn toàn.

Nên script này:
  1. tải về theo từng notebook, không khử trùng lặp;
  2. tính accuracy từng file rồi đối chiếu Bảng 1 của bài;
  3. chỉ gom vào results_table1/ những file khớp Bảng 1 trong 0,05 điểm.

Cấu hình nào không có bản khớp thì báo thiếu, chứ không lấy đại bản gần nhất.

    python fetch_from_kaggle.py
"""

import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "results_by_notebook"
MATCHED = HERE / "results_table1"

# Mọi notebook có khả năng chứa kết quả train, kiểm kê 16/11/2026.
KERNELS = [
    # offline augment (tháng 8) — nghi là nguồn của Bảng 1
    "minhhoang17/waste-36-offline-v2",
    "minhhoang17/waste-36-offline-augment",
    "minhhoang17/waste-36-labels-28-06",
    # online augment (tháng 6)
    "minhhoang17/36-labels-gpu-with-augment-8-6-2026",
    "minhhoang17/36labe-gpu-with-augment-9-6-2026-from-seed36-dense",
    "minhhoang17/36labe-gpu-with-augment9-6-2026-from-seed42-resnet",
    # các lần sớm hơn (tháng 3)
    "minhhoang17/waste-36-labels-gpu-with-augment",
    "minhhoang17/waste-36-labels-tpu-with-augment",
    "minhhoang17/waste-36-labels-gpu-no-augment",
    "minhhoang17/waste-36-labels-tpu-no-augment",
]

FILE_PATTERN = r"TEST_seed\d+_.+_y_(pred|prob|true)\.npy"
NAME_RE = re.compile(r"^TEST_(seed\d+)_(.+)_y_(pred|prob|true)\.npy$")

# Bảng 1 của bài (accuracy test, %). Đây là mốc nhận dạng lần chạy.
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
TOL = 0.05  # Bảng 1 ghi 2 chữ số nên sai số làm tròn chỉ cỡ 0,005


def get_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Chưa cài gói kaggle: pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        sys.exit(f"Chưa xác thực Kaggle: {e}\nChạy: kaggle auth login")
    return api


def pull(api, kernel, dest):
    token, total = None, 0
    while True:
        try:
            files, token = api.kernels_output(
                kernel, path=str(dest), file_pattern=FILE_PATTERN,
                force=False, quiet=True, page_token=token, page_size=200,
            )
        except Exception as e:
            print(f"  lỗi: {str(e)[:120]}")
            return total
        total += len(files)
        if not token:
            return total


def main():
    api = get_api()
    RAW.mkdir(exist_ok=True)
    MATCHED.mkdir(exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix="kg_"))
    # (kernel, seed, model) -> (acc, đường dẫn npz)
    inventory = {}

    try:
        for kernel in KERNELS:
            slug = kernel.split("/")[1]
            print(f"{slug:<52} ", end="", flush=True)
            n = pull(api, kernel, staging / slug)
            print(f"{n:>3} file")

            parts = defaultdict(dict)
            for f in (staging / slug).rglob("*.npy"):
                m = NAME_RE.match(f.name)
                if m:
                    seed, model, kind = m.groups()
                    parts[(seed, model)].setdefault(kind, f)

            out_dir = RAW / slug
            for (seed, model), kinds in parts.items():
                if len(kinds) < 3:
                    continue
                y_pred = np.load(kinds["pred"])
                y_prob = np.load(kinds["prob"])
                y_true = np.load(kinds["true"])
                if not (len(y_pred) == len(y_true) == len(y_prob)):
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                p = out_dir / f"preds_{seed}_{model}.npz"
                np.savez_compressed(
                    p,
                    y_pred=y_pred.astype(np.int16),
                    y_prob=y_prob.astype(np.float32),
                    y_true=y_true.astype(np.int16),
                )
                inventory[(slug, seed, model)] = (
                    float((y_pred == y_true).mean() * 100), p
                )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # ------------------------------------------------ đối chiếu Bảng 1
    print("\n" + "=" * 78)
    print("CẤU HÌNH NÀO KHỚP BẢNG 1 (sai số ≤ 0,05 điểm)")
    print("=" * 78)

    per_kernel = defaultdict(lambda: [0, 0])  # slug -> [khớp, tổng]
    for (slug, seed, model), (acc, _) in inventory.items():
        per_kernel[slug][1] += 1
        if model in TABLE1 and abs(acc - TABLE1[model]) <= TOL:
            per_kernel[slug][0] += 1

    for slug in sorted(per_kernel, key=lambda s: -per_kernel[s][0]):
        hit, tot = per_kernel[slug]
        print(f"  {slug:<52} khớp {hit:>2}/{tot:>2}")

    # gom bản khớp
    print("\n" + "=" * 78)
    print("GOM BẢN KHỚP VÀO results_table1/")
    print("=" * 78)

    chosen, ambiguous = {}, defaultdict(list)
    for (slug, seed, model), (acc, p) in inventory.items():
        if model in TABLE1 and abs(acc - TABLE1[model]) <= TOL:
            ambiguous[model].append((slug, seed, acc, p))

    for model in TABLE1:
        cands = ambiguous.get(model, [])
        if not cands:
            print(f"  [THIẾU] {model:<34} (Bảng 1: {TABLE1[model]:.2f})")
            continue
        # ưu tiên bản khớp sát nhất, không phải bản tìm thấy trước:
        # v1 và v2 chênh nhau cỡ 0,03 điểm nên "tìm thấy trước" dễ chọn nhầm
        # v2 trong khi Bảng 1 là v1, và trộn hai lần chạy là điều phải tránh.
        cands.sort(key=lambda c: abs(c[2] - TABLE1[model]))
        slug, seed, acc, p = cands[0]
        shutil.copy(p, MATCHED / f"preds_{model}.npz")
        chosen[model] = (slug, seed, acc)
        extra = f"  [+{len(cands)-1} bản trùng]" if len(cands) > 1 else ""
        print(f"  [OK]    {model:<34} {acc:6.2f}  ←  {slug} / {seed}{extra}")

    print(f"\nGom được {len(chosen)}/18 cấu hình.")

    if len(chosen) == 18:
        seeds = {s for _, s, _ in chosen.values()}
        srcs = {k for k, _, _ in chosen.values()}
        print(f"  seed: {', '.join(sorted(seeds))}")
        print(f"  notebook nguồn: {', '.join(sorted(srcs))}")
        print("\nChạy tiếp:")
        print("    python analysis_02_diversity.py --dir results_table1")
    else:
        missing = [m for m in TABLE1 if m not in chosen]
        print("\n  Thiếu: " + ", ".join(missing))
        print("  => Dự đoán của những cấu hình này chưa từng được lưu, hoặc nằm")
        print("     ở notebook chưa quét. Xem results_by_notebook/ để đối chiếu,")
        print("     hoặc chạy lại inference bằng kaggle_01_dump_predictions.py.")


if __name__ == "__main__":
    main()
