"""Tải dự đoán trên tập TRAIN để tính độ chênh train-test.

Chỉ lấy y_pred và y_true (không lấy y_prob, vì phần này chỉ cần accuracy),
nên nhẹ hơn nhiều. Dùng đúng ba notebook của bộ online 8-9/6.
"""

import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "train_acc.csv"

KERNELS = [
    "minhhoang17/36-labels-gpu-with-augment-8-6-2026",
    "minhhoang17/36labe-gpu-with-augment-9-6-2026-from-seed36-dense",
    "minhhoang17/36labe-gpu-with-augment9-6-2026-from-seed42-resnet",
]
PATTERN = r"TRAIN_seed\d+_.+_y_(pred|true)\.npy"
NAME_RE = re.compile(r"^TRAIN_(seed\d+)_(.+)_y_(pred|true)\.npy$")


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    staging = Path(tempfile.mkdtemp(prefix="kgtr_"))
    rows = []
    try:
        for kernel in KERNELS:
            slug = kernel.split("/")[1]
            print(f"{slug:<52} ", end="", flush=True)
            token, n = None, 0
            while True:
                files, token = api.kernels_output(
                    kernel, path=str(staging / slug), file_pattern=PATTERN,
                    force=False, quiet=True, page_token=token, page_size=200,
                )
                n += len(files)
                if not token:
                    break
            print(f"{n:>3} file")

            parts = defaultdict(dict)
            for f in (staging / slug).rglob("*.npy"):
                m = NAME_RE.match(f.name)
                if m:
                    seed, model, kind = m.groups()
                    parts[(seed, model)].setdefault(kind, f)

            for (seed, model), kinds in parts.items():
                if len(kinds) < 2:
                    continue
                yp, yt = np.load(kinds["pred"]), np.load(kinds["true"])
                if len(yp) != len(yt):
                    continue
                rows.append({
                    "seed": seed, "model": model, "n_train": len(yt),
                    "train_acc": round(float((yp == yt).mean() * 100), 2),
                    "source": slug,
                })
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    import csv
    seen = set()
    uniq = []
    for r in sorted(rows, key=lambda r: (r["seed"], r["model"])):
        k = (r["seed"], r["model"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)

    by_seed = defaultdict(int)
    for r in uniq:
        by_seed[r["seed"]] += 1
    print(f"\nGhi {len(uniq)} dòng vào {OUT.name}")
    for s in sorted(by_seed):
        print(f"  {s}: {by_seed[s]} cấu hình, n_train={uniq[0]['n_train']}")


if __name__ == "__main__":
    main()
