"""
Bước 1 — Xuất dự đoán theo từng ảnh cho cả 18 cấu hình.

Chạy trên Kaggle (GPU P100). Mỗi model chỉ chạy inference, KHÔNG train lại.
Ước tính: 20-40 phút cho cả 18 model trên 3.552 ảnh test.

Đầu ra: results/preds_<backbone>_<head>.npz, mỗi file chứa
    y_pred  (N,)      nhãn dự đoán
    y_prob  (N, 36)   xác suất softmax
    y_true  (N,)      nhãn thật
và results/test_manifest.csv — danh sách đường dẫn ảnh theo đúng thứ tự.

Thứ tự ảnh là thứ nguy hiểm nhất ở đây: mọi phép so từng cặp model sau này
chỉ đúng nếu cả 18 file dùng chung một thứ tự. Script cố định shuffle=False,
lưu manifest, và đối chiếu accuracy vừa tính với Bảng 1 của bài. Lệch quá
ngưỡng là dừng, không ghi file.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------- cấu hình

# Sửa hai đường dẫn này cho khớp Kaggle input của bạn.
MODEL_DIR = Path("/kaggle/input/waste-models-90-10/models-90-10")
TEST_DIR = Path("/kaggle/input/waste-36-stratified/test")

OUT_DIR = Path("/kaggle/working/results")
BATCH_SIZE = 32

# Kích thước đầu vào theo backbone (mục 4.2 của bài).
INPUT_SIZE = {
    "efficient_net": 224,
    "mobilenet_v3": 224,
    "resnet_50": 224,
    "dense_net": 224,
    "inception_v3": 299,
    "xception": 299,
}

HEADS = ["SIMPLE_FC", "GAP_BN_FC_Drop", "DUAL_POOLING_FC"]

# Accuracy test trong Bảng 1, dùng làm mốc đối chiếu. Nếu dự đoán vừa tính
# lệch khỏi mốc này quá ACC_TOL thì gần như chắc chắn sai thứ tự ảnh hoặc
# sai tiền xử lý — khi đó dừng hẳn, vì mọi phân tích sau đó sẽ vô nghĩa.
REFERENCE_ACC = {
    ("efficient_net", "SIMPLE_FC"): 90.51,
    ("efficient_net", "GAP_BN_FC_Drop"): 92.03,
    ("efficient_net", "DUAL_POOLING_FC"): 92.12,
    ("dense_net", "SIMPLE_FC"): 91.08,
    ("dense_net", "GAP_BN_FC_Drop"): 92.20,
    ("dense_net", "DUAL_POOLING_FC"): 91.89,
    ("xception", "SIMPLE_FC"): 90.32,
    ("xception", "GAP_BN_FC_Drop"): 91.10,
    ("xception", "DUAL_POOLING_FC"): 90.60,
    ("resnet_50", "SIMPLE_FC"): 89.39,
    ("resnet_50", "GAP_BN_FC_Drop"): 90.77,
    ("resnet_50", "DUAL_POOLING_FC"): 90.57,
    ("inception_v3", "SIMPLE_FC"): 88.40,
    ("inception_v3", "GAP_BN_FC_Drop"): 90.03,
    ("inception_v3", "DUAL_POOLING_FC"): 88.82,
    ("mobilenet_v3", "SIMPLE_FC"): 84.91,
    ("mobilenet_v3", "GAP_BN_FC_Drop"): 87.95,
    ("mobilenet_v3", "DUAL_POOLING_FC"): 87.75,
}

ACC_TOL = 0.75  # điểm phần trăm


# ------------------------------------------------------------------ dữ liệu


def build_test_ds(image_size):
    """Nạp tập test theo thứ tự tất định.

    image_dataset_from_directory với shuffle=False sắp xếp theo tên lớp rồi
    tên file, nên thứ tự tái lập được giữa các lần chạy và giữa các model.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=(image_size, image_size),
        shuffle=False,
        interpolation="bilinear",
    )
    class_names = list(ds.class_names)
    file_paths = list(ds.file_paths)
    return ds.prefetch(tf.data.AUTOTUNE), class_names, file_paths


def collect_labels(ds):
    return np.concatenate([y.numpy() for _, y in ds], axis=0)


# ------------------------------------------------------------------- chạy


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dựng sẵn hai dataset, một cho 224 và một cho 299, để khỏi nạp lại
    # mỗi model. Nhãn và manifest lấy từ bản 224; bản 299 phải khớp y hệt.
    ds224, class_names, paths224 = build_test_ds(224)
    ds299, class_names299, paths299 = build_test_ds(299)

    assert class_names == class_names299, "Thứ tự lớp giữa hai dataset khác nhau"
    assert paths224 == paths299, "Thứ tự ảnh giữa hai dataset khác nhau"

    y_true = collect_labels(ds224)
    y_true299 = collect_labels(ds299)
    assert np.array_equal(y_true, y_true299), "Nhãn giữa hai dataset khác nhau"

    n = len(y_true)
    print(f"Tập test: {n} ảnh, {len(class_names)} lớp")

    manifest = pd.DataFrame(
        {
            "index": np.arange(n),
            "path": paths224,
            "label": y_true,
            "class_name": [class_names[i] for i in y_true],
        }
    )
    manifest.to_csv(OUT_DIR / "test_manifest.csv", index=False)
    (OUT_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))

    results, failures = [], []

    for backbone, size in INPUT_SIZE.items():
        ds = ds224 if size == 224 else ds299
        for head in HEADS:
            name = f"{backbone}_{head}"
            model_path = MODEL_DIR / f"model_seed36_{name}.keras"

            if not model_path.exists():
                print(f"[THIẾU] {model_path.name}")
                failures.append((name, "không tìm thấy file"))
                continue

            print(f"[CHẠY ] {name} ... ", end="", flush=True)
            model = tf.keras.models.load_model(model_path, compile=False)
            y_prob = model.predict(ds, verbose=0)
            del model
            tf.keras.backend.clear_session()

            if y_prob.shape[0] != n:
                failures.append((name, f"số dòng {y_prob.shape[0]} != {n}"))
                print("LỆCH SỐ DÒNG")
                continue

            y_pred = y_prob.argmax(axis=1)
            acc = float((y_pred == y_true).mean() * 100)
            ref = REFERENCE_ACC[(backbone, head)]
            delta = acc - ref

            status = "OK" if abs(delta) <= ACC_TOL else "LỆCH"
            print(f"acc {acc:.2f}% (Bảng 1: {ref:.2f}%, lệch {delta:+.2f}) [{status}]")

            if status == "LỆCH":
                failures.append((name, f"acc {acc:.2f} vs {ref:.2f}"))
                continue

            np.savez_compressed(
                OUT_DIR / f"preds_{name}.npz",
                y_pred=y_pred.astype(np.int16),
                y_prob=y_prob.astype(np.float32),
                y_true=y_true.astype(np.int16),
            )
            results.append(
                {
                    "model": name,
                    "backbone": backbone,
                    "head": head,
                    "acc": acc,
                    "acc_table1": ref,
                    "delta": delta,
                }
            )

    pd.DataFrame(results).to_csv(OUT_DIR / "reproduced_accuracy.csv", index=False)

    print("\n" + "=" * 62)
    print(f"Thành công: {len(results)}/18")
    if failures:
        print("Có vấn đề:")
        for name, why in failures:
            print(f"  - {name}: {why}")
        print(
            "\nAccuracy lệch khỏi Bảng 1 gần như luôn là một trong hai nguyên nhân:\n"
            "  (1) thứ tự ảnh khác lúc train — kiểm tra lại cách dựng tập test;\n"
            "  (2) tiền xử lý bị áp hai lần hoặc thiếu — các model này nhúng\n"
            "      preprocess_input bên trong, nên đầu vào phải là RGB thô [0,255].\n"
            "Đừng chạy bước 2 khi chưa khắc phục: mọi số liệu sau đó sẽ sai."
        )
    else:
        print("Cả 18 model khớp Bảng 1. Sang bước 2 được.")


if __name__ == "__main__":
    main()
