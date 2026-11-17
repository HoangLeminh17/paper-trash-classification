"""Đường dẫn dùng chung: chạy được cả ở thư mục làm việc lẫn trong repo công khai.

Repo công khai:  predictions/{test,val}_<seed>/, kết quả -> results/, hình -> figures/
Thư mục làm việc: results_online_<seed>/, results_val_<seed>/, kết quả cạnh script.
"""
from pathlib import Path

HERE = Path(__file__).parent
IN_REPO = (HERE.parent / "predictions").is_dir()

OUT = HERE.parent / "results" if IN_REPO else HERE
FIGDIR = HERE.parent / "figures" if IN_REPO else HERE.parent / "Paper-IDA2027" / "figures"


def test_dir(seed):
    return HERE.parent / "predictions" / f"test_{seed}" if IN_REPO else HERE / f"results_online_{seed}"


def val_dir(seed):
    return HERE.parent / "predictions" / f"val_{seed}" if IN_REPO else HERE / f"results_val_{seed}"
