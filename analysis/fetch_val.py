"""Tải dự đoán trên tập VALIDATION.

Cần cho việc bịt lỗ rò tập test: muốn nói quy tắc chọn thành viên dùng được
trong thực tế thì phải chọn bằng dữ liệu không phải test.
"""

import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
KERNELS = [
    "minhhoang17/36-labels-gpu-with-augment-8-6-2026",
    "minhhoang17/36labe-gpu-with-augment-9-6-2026-from-seed36-dense",
    "minhhoang17/36labe-gpu-with-augment9-6-2026-from-seed42-resnet",
]
PATTERN = r"VAL_seed\d+_.+_y_(pred|prob|true)\.npy"
NAME_RE = re.compile(r"^VAL_(seed\d+)_(.+)_y_(pred|prob|true)\.npy$")

BACKBONES = ["efficient_net", "mobilenet_v3", "resnet_50",
             "dense_net", "inception_v3", "xception"]
HEADS = ["SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"]
WANTED = {f"{b}_{h}" for b in BACKBONES for h in HEADS}


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    staging = Path(tempfile.mkdtemp(prefix="kgval_"))
    parts = defaultdict(dict)
    try:
        for kernel in KERNELS:
            slug = kernel.split("/")[1]
            print(f"{slug:<52} ", end="", flush=True)
            token, n = None, 0
            while True:
                files, token = api.kernels_output(
                    kernel, path=str(staging / slug), file_pattern=PATTERN,
                    force=False, quiet=True, page_token=token, page_size=200)
                n += len(files)
                if not token:
                    break
            print(f"{n:>3} file")
            for f in (staging / slug).rglob("*.npy"):
                m = NAME_RE.match(f.name)
                if m:
                    seed, model, kind = m.groups()
                    parts[(seed, model)].setdefault(kind, f)

        by_seed = defaultdict(list)
        for (seed, model), kinds in sorted(parts.items()):
            if len(kinds) < 3:
                continue
            yp, yb, yt = (np.load(kinds[k]) for k in ("pred", "prob", "true"))
            if not (len(yp) == len(yt) == len(yb)):
                continue
            out = HERE / f"results_val_{seed}"
            out.mkdir(exist_ok=True)
            np.savez_compressed(out / f"preds_{model}.npz",
                                y_pred=yp.astype(np.int16),
                                y_prob=yb.astype(np.float32),
                                y_true=yt.astype(np.int16))
            by_seed[seed].append(model)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print()
    for seed in sorted(by_seed):
        have = set(by_seed[seed])
        miss = WANTED - have
        n = len(np.load(HERE / f"results_val_{seed}" /
                        f"preds_{sorted(have)[0]}.npz")["y_true"])
        print(f"  {seed}: {len(have)}/18" + (" ĐỦ" if not miss else
              f" thiếu {len(miss)}") + f", n_val={n}")


if __name__ == "__main__":
    main()
