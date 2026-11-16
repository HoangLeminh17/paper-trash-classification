# Phân tích độ đa dạng — chuẩn bị bản IDA 2027

Mục tiêu: biến bài từ "so sánh 18 cấu hình" thành "độ đa dạng, chứ không phải
độ chính xác, quyết định lợi ích ensemble" — luận điểm hợp gu IDA hơn hẳn.

## Kiểm kê Kaggle (dò ngày 16/11/2026)

Tin tốt: **không cần chạy lại inference.** Các notebook cũ đã lưu sẵn
`y_pred` / `y_prob` / `y_true` theo từng ảnh. Thư mục local chỉ có confusion
matrix (đã gộp theo lớp, vô dụng cho phân tích đa dạng), nhưng Kaggle thì có
đủ bản chi tiết.

| Notebook | Nội dung |
|---|---|
| `36-labels-gpu-with-augment-8-6-2026` | seed36: efficient_net, inception_v3, mobilenet_v3, resnet_50, xception — 15 cấu hình |
| `36labe-gpu-with-augment-9-6-2026-from-seed36-dense` | seed36: dense_net (3) + seed42: 13 cấu hình |
| `36labe-gpu-with-augment9-6-2026-from-seed42-resnet` | seed42: dense_net, resnet_50 (6) + seed123: 10 cấu hình |
| `waste-36-offline-v2`, `waste-36-offline-augment`, `waste-36-labels-28-06` | các lần chạy offline-augment, 4–6 cấu hình mỗi notebook |

Cộng lại: **seed36 đủ 18/18, seed42 đủ 18/18, seed123 được 10/18.**

Tức là bài có 2 seed đầy đủ chứ không phải 1 như thư mục local gợi ý — câu
"multi-seed runs" trong contributions có số liệu đỡ lưng, chỉ cần lấy về.

Tổng dung lượng phần cần tải khoảng 10 MB (`y_prob` 512 kB/model), không đụng
tới file `.keras` nặng vài GB.

## Chạy

**Bước 0 — tải về**

Xác thực một lần, không phải giữ khoá nào:

```
kaggle auth login
```

Rồi:

```
python fetch_from_kaggle.py
```

Script dùng cơ chế xác thực của chính gói `kaggle`, nên chạy được với cả
OAuth, `~/.kaggle/access_token`, biến `KAGGLE_API_TOKEN`, lẫn `kaggle.json`
kiểu cũ. Nó lọc bằng `file_pattern` nên chỉ tải `.npy`, không đụng `.keras`.

Script lọc đúng file `.npy` cần, gộp thành `results/<seed>/preds_<model>.npz`,
và in ra seed nào đủ seed nào thiếu.

**Bước 1 — chỉ khi bước 0 thiếu cấu hình nào đó**

`kaggle_01_dump_predictions.py` chạy lại inference từ 18 model `.keras` còn
lưu trên máy. Script tự đối chiếu accuracy với Bảng 1; lệch quá 0,75 điểm thì
không ghi file.

**Bước 2 — phân tích (CPU, vài phút)**

```
pip install numpy pandas scipy scikit-learn requests
python analysis_02_diversity.py --dir results/seed36
python analysis_02_diversity.py --dir results/seed42
```

Chạy cả hai seed rồi đối chiếu: kết luận nào giữ nguyên qua cả hai seed thì
mới đem vào bài.

## Bẫy đã chặn sẵn

**Thứ tự ảnh.** Mọi phép so từng cặp chỉ đúng nếu cả 18 file dùng chung một
thứ tự ảnh. Bước 1 cố định `shuffle=False`, lưu `test_manifest.csv`, và kiểm
tra chéo giữa dataset 224 và 299. Bước 2 từ chối chạy nếu vector nhãn thật
giữa các file không trùng nhau.

**Tiền xử lý.** Các model nhúng `preprocess_input` bên trong, nên đầu vào phải
là RGB thô [0,255]. Áp thêm một lần nữa ở ngoài sẽ làm accuracy tụt mạnh —
phép đối chiếu với Bảng 1 bắt được lỗi này ngay.

## Đầu ra và ý nghĩa

| File | Nội dung | Dùng để |
|---|---|---|
| `pairwise.csv` | 153 cặp: bất đồng, Q-statistic, kappa, double-fault, McNemar | Chứng minh accuracy ngang nhau ≠ sai giống nhau |
| `triples.csv` | 816 tổ hợp: accuracy ensemble, lợi ích, chỉ số đa dạng | Hình trung tâm của bài |
| `strategies.csv` | Top-3 accuracy vs bất đồng cao nhất vs ngẫu nhiên | Trả lời "chọn đa dạng có thắng chọn giỏi không" |
| `regression.txt` | Lợi ích ~ accuracy trung bình + độ bất đồng | Độ đa dạng còn ý nghĩa sau khi kiểm soát accuracy? |
| `summary.txt` | Tóm tắt bằng lời | Đọc trước khi viết |

Kết quả quyết định hướng viết bài: nếu hệ số của độ bất đồng vẫn có ý nghĩa
sau khi đã kiểm soát accuracy trung bình thì luận điểm đứng được. Nếu không,
phải đổi góc — và biết điều đó trước khi viết vẫn hơn là để reviewer phát hiện.

## Giao thức trung thực (quan trọng nhất)

`analysis_02_diversity.py` đo độ đa dạng trên chính tập test, nên quy tắc chọn
rút ra từ nó có rò thông tin test. `analysis_03_honest.py` làm lại cho sạch:
chọn thành viên **chỉ bằng tập validation**, rồi mới chấm trên test.

Kết quả khác nhau rõ, và bài dùng bản trung thực:

| | đo trên test (02) | đo trên val (03) |
|---|---|---|
| Bất đồng cao nhất trong nhóm ±2 điểm | +3,72 / +3,27 | **+1,69 / +2,08** |
| Mỗi backbone một model | +2,62 / +2,51 | **+1,86 / +2,76** |

Quy tắc tối đa hoá đa dạng **không sống sót** — khoảng một nửa lợi thế của nó
là do khớp tập test. Quan hệ thống kê thì vẫn còn (R² 0,29 / 0,37, hệ số bất
đồng +43,3 / +42,0 với biến đo trên val), nhưng một tín hiệu thống kê thật
không tự động cho ra một quy tắc chọn đáng tin.

Thứ sống sót là quy tắc cấu trúc: mỗi backbone một model, và dùng k = 4–5
thay vì 3.

## Thứ tự chạy

```
kaggle auth login
python fetch_from_kaggle.py     # test, giữ nguyên nguồn gốc từng notebook
python fetch_val.py             # validation
python fetch_train.py           # train, chỉ để tính độ chênh
python assemble_online.py       # dựng bộ 18 cấu hình theo seed
python analysis_03_honest.py    # phân tích chính của bài
python paper_tables.py          # bảng
python make_figures.py          # 4 hình
```

## Việc còn lại

- seed123 mới 10/18. Chạy nốt 8 cấu hình thì có seed thứ ba.
- Dataset thứ hai (TrashNet) để kiểm tra kết luận không chỉ đúng với riêng rác.
- Tập validation chỉ 3.511 ảnh; val lớn hơn có thể cứu được quy tắc
  tối đa hoá đa dạng. Chưa thử.
