# FPT Stock Forecasting

Dự án dự báo giá đóng cửa tiếp theo của cổ phiếu FPT bằng các mô hình Deep Learning cho chuỗi thời gian. Hệ thống bao gồm pipeline thu thập dữ liệu, xây dựng dataset, huấn luyện mô hình, lưu artifact và demo dự báo bằng Streamlit.

## Mục tiêu dự án

Dự án tập trung vào bài toán dự báo `next close`, tức là dùng dữ liệu giao dịch các ngày trước để dự báo giá đóng cửa của phiên kế tiếp.

Các mô hình chính được sử dụng:

- LSTM
- CNN1D-LSTM 
- Transformer
- Informer
- Autoformer

Hệ thống có hai hướng dữ liệu:

- **Raw FPT only**: chỉ dùng dữ liệu gốc `open`, `high`, `low`, `close`, `volume`.
- **Merged dataset**: dùng thêm các đặc trưng kỹ thuật và dữ liệu VNIndex để phục vụ thí nghiệm nội bộ.

Demo Streamlit hiện ưu tiên dùng dataset **raw FPT**.

## Cấu trúc thư mục

```
FPT-STOCK-FORECASTING/
├── 📁 .github
│   └── 📁 workflows
│       ├── ⚙️ daily_crawl.yml
│       ├── ⚙️ monthly_retrain.yml
│       └── ⚙️ weekly_evaluate.yml
├── 📁 app
│   └── 🐍 streamlit_app.py
├── 📁 artifacts
│   └── 📁 raw_fpt_only_residual_cnnlstm_transformer
│       ├── 📁 output
├── 📁 data
│   ├── 📁 processed/
│   └── 📁 raw/
├── 📁 data_pipeline
│   ├── 🐍 base.py
│   ├── 🐍 build_dataset.py
│   ├── 🐍 crawl_fpt.py
│   ├── 🐍 crawl_vnindex.py
│   └── 🐍 daily_crawl.py
├── 📁 notebooks
│   ├── 📄 data_eda_visualize.ipynb
│   ├── 📄 fpt_stock_merged_dataset_residual_cnnlstm_transformer.ipynb
│   └── 📄 fpt_stock_raw_fpt_residual_cnnlstm_transformer.ipynb
├── 📁 training
│   ├── 🐍 config.py
│   ├── 🐍 data.py
│   ├── 🐍 evaluate.py
│   ├── 🐍 metrics.py
│   ├── 🐍 models.py
│   ├── 🐍 predict.py
│   ├── 🐍 retrain.py
│   ├── 🐍 train.py
│   └── 🐍 utils.py
├── ⚙️ .gitignore
├── 📝 README.md
├── 🐍 main.py
├── ⚙️ pyproject.toml
└── 📄 uv.lock
```

## Ý nghĩa các thư mục chính

| Thư mục | Vai trò |
|---|---|
| `data_pipeline/` | Crawl dữ liệu FPT, VNIndex và build dataset |
| `data/raw/` | Lưu dữ liệu gốc sau khi crawl |
| `data/processed/` | Lưu dataset đã xử lý: `merged_dataset.csv` |
| `notebooks/` | Thí nghiệm, huấn luyện và đánh giá mô hình |
| `training/` | Code Python dùng lại cho train, predict, metrics, models |
| `artifacts/` | Lưu model weights, scaler, feature columns và metrics |
| `app/` | Demo Streamlit cho người dùng upload CSV và dự báo |

## Dữ liệu đầu vào

### Raw FPT only

Dataset raw chỉ cần các cột:

```text
time, open, high, low, close, volume
```

Ví dụ:

```csv
time,open,high,low,close,volume
2026-06-12,74.0,75.0,73.5,74.2,1234567
2026-06-15,74.4,75.2,73.5,73.6,6422700
```

### Merged dataset

Merged dataset dùng cho thí nghiệm nội bộ, gồm OHLCV, VNIndex và các đặc trưng kỹ thuật như MA, RSI, MACD, ATR, volatility, volume ratio. File này không được dùng làm input chính cho demo người dùng vì không đảm bảo dữ liệu bên ngoài có cùng bộ feature.

## Pipeline tổng quát

```text
Raw CSV / Uploaded CSV
        ↓
Clean dữ liệu, sort theo time tăng dần
        ↓
Giữ đúng các feature cần thiết
        ↓
Scale input bằng feature_scaler.pkl
        ↓
Tạo sliding window
        ↓
Load model architecture + checkpoint .pt
        ↓
Predict next close
        ↓
Inverse transform bằng target_scaler.pkl
        ↓
Hiển thị kết quả và biểu đồ trên Streamlit
```

## Artifact sau khi train

Sau khi huấn luyện, notebook hoặc script training sẽ lưu artifact vào thư mục `artifacts/`.

Ví dụ với raw model:

```text
artifacts/raw_fpt_only_residual_cnnlstm_transformer/
│
├── best_lstm_w3.pt
├── best_cnn1d_lstm_w5.pt
├── best_transformer_w7.pt
├── best_informer_w30.pt
├── best_autoformer_w3.pt
├── feature_scaler.pkl
├── target_scaler.pkl
├── feature_columns.json
├── metrics_dev.csv
├── metrics_test.csv
└── metrics_all_windows_dev_test.csv
```

Ý nghĩa:

| File | Vai trò |
|---|---|
| `best_*.pt` | Trọng số model PyTorch đã train |
| `feature_scaler.pkl` | Scaler chuẩn hóa input giống lúc train |
| `target_scaler.pkl` | Scaler đưa prediction về giá thật |
| `feature_columns.json` | Thứ tự feature đưa vào model |
| `metrics_*.csv` | Kết quả đánh giá model |

## Cài đặt môi trường

Dự án dùng `uv` để quản lý môi trường Python.

```bash
uv sync
```

Nếu chưa có `uv`, có thể cài bằng:

```bash
pip install uv
```

## Chạy pipeline dữ liệu

Crawl hoặc cập nhật dữ liệu hằng ngày:

```bash
uv run python -m data_pipeline.daily_crawl
```

Build lại merged dataset:

```bash
uv run python -m data_pipeline.build_dataset
```

## Huấn luyện mô hình

Train với raw FPT only:

```bash
uv run python -m training.train --mode raw --data data/raw/fpt_stock_price.csv
```

Train với merged dataset:

```bash
uv run python -m training.train --mode merged --data data/processed/merged_dataset.csv
```

Chạy nhanh để test code:

```bash
uv run python -m training.train --mode raw --data data/raw/fpt_stock_price.csv --epochs 3 --windows 3 --models lstm cnn1d_lstm
```

## Chạy demo Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

Sau khi chạy, mở trình duyệt theo địa chỉ Streamlit hiển thị

## Cách dùng demo

1. Chạy Streamlit app.
2. Upload file CSV lịch sử giao dịch.
3. File CSV chỉ cần có các cột `time`, `open`, `high`, `low`, `close`, `volume`.
4. Chọn model checkpoint hoặc để hệ thống tự chọn model tốt nhất theo RMSE.
5. App sẽ hiển thị:
   - Giá đóng cửa hiện tại.
   - Giá đóng cửa dự báo cho phiên kế tiếp.
   - Mức thay đổi dự báo.
   - Biểu đồ so sánh giá thực tế và giá dự báo.
   - Bảng dự báo theo ngày, sắp xếp ngày mới nhất lên đầu.
6. Có thể nhập tay một dòng OHLCV mới để dự báo phiên kế tiếp.

## Metrics đánh giá

Các chỉ số đánh giá chính:

- MAE
- RMSE
- MAPE
- R2
- Directional Accuracy
- Bias

Trong đó RMSE thường được dùng để chọn checkpoint tốt nhất.

## Lưu ý khi deploy

Demo Streamlit hiện dùng artifact raw:

```text
artifacts/raw_fpt_only_residual_cnnlstm_transformer
```

## Tóm tắt

Đây là một project Deep Learning end-to-end cho bài toán dự báo giá cổ phiếu FPT, bao gồm:

```text
Data crawling → Dataset building → Model training → Artifact saving → Streamlit demo → Next-close prediction
```

