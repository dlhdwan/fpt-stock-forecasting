from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
import torch


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from training.predict import (
    checkpoint_display_name,
    clean_stock_data,
    format_result_table,
    list_checkpoints,
    load_model_bundle,
    predict_from_dataframe,
    read_uploaded_csv,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "raw_fpt_only_residual_cnnlstm_transformer"


st.set_page_config(
    page_title="FPT Stock Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource(show_spinner="Đang tải model...")
def cached_load_model_bundle(artifact_dir_text: str, checkpoint_name: str):
    artifact_dir = Path(artifact_dir_text)
    checkpoint_path = artifact_dir / checkpoint_name

    return load_model_bundle(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        device=DEVICE,
    )


@st.cache_data(show_spinner=False)
def cached_read_csv(file_bytes: bytes) -> pd.DataFrame:
    return read_uploaded_csv(file_bytes)


def show_example_csv() -> None:
    example_df = pd.DataFrame(
        {
            "time": ["2026-06-10", "2026-06-11", "2026-06-12"],
            "open": [118000, 119000, 120000],
            "high": [120000, 121000, 122000],
            "low": [117000, 118000, 119000],
            "close": [119000, 120000, 121000],
            "volume": [2500000, 2700000, 2600000],
        }
    )

    st.info("CSV cần có tối thiểu các cột: time, open, high, low, close, volume.")
    st.dataframe(example_df, width="stretch")


def build_chart(result_df: pd.DataFrame):
    chart_df = result_df.copy()

    chart_df = chart_df.dropna(
        subset=["predict_for_date", "actual_next_close", "predicted_next_close"]
    ).copy()

    if chart_df.empty:
        return None

    chart_df["Ngày"] = pd.to_datetime(chart_df["predict_for_date"])
    chart_df = chart_df.sort_values("Ngày", ascending=True)

    chart_df = chart_df.rename(
        columns={
            "actual_next_close": "Giá thực tế",
            "predicted_next_close": "Giá dự báo",
        }
    )

    chart_df = chart_df.set_index("Ngày")

    return chart_df[["Giá thực tế", "Giá dự báo"]]


def main() -> None:
    st.title("📈 Dự báo giá đóng cửa cổ phiếu FPT")

    st.caption(
        "Demo sử dụng model đã train trên dữ liệu raw OHLCV. "
        "CSV upload chỉ cần các cột: time, open, high, low, close, volume. "
        "Nếu CSV có thêm cột khác, hệ thống sẽ tự bỏ qua."
    )

    if not ARTIFACT_DIR.exists():
        st.error(f"Không tìm thấy artifact folder: {ARTIFACT_DIR}")
        st.stop()

    checkpoints = list_checkpoints(ARTIFACT_DIR)

    if not checkpoints:
        st.error(f"Không tìm thấy checkpoint best_*.pt trong: {ARTIFACT_DIR}")
        st.stop()

    checkpoint_names = [path.name for path in checkpoints]
    checkpoint_labels = [checkpoint_display_name(path) for path in checkpoints]

    with st.sidebar:
        st.header("⚙️ Cấu hình dự báo")

        selected_label = st.selectbox(
            "Chọn model checkpoint",
            options=checkpoint_labels,
            index=0,
        )

        selected_index = checkpoint_labels.index(selected_label)
        selected_checkpoint_name = checkpoint_names[selected_index]

        max_rows = st.slider(
            "Số dòng gần nhất để hiển thị",
            min_value=30,
            max_value=1000,
            value=300,
            step=10,
        )

        st.caption(f"Device: {DEVICE}")
        st.caption(f"Checkpoint: {selected_checkpoint_name}")

    bundle = cached_load_model_bundle(
        artifact_dir_text=str(ARTIFACT_DIR),
        checkpoint_name=selected_checkpoint_name,
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV dữ liệu FPT",
        type=["csv"],
    )

    if uploaded_file is None:
        show_example_csv()
        st.stop()

    try:
        file_bytes = uploaded_file.getvalue()
        raw_df = cached_read_csv(file_bytes)
        clean_df = clean_stock_data(raw_df)
    except Exception as error:
        st.error(f"Lỗi đọc hoặc clean CSV: {error}")
        st.stop()

    st.subheader("📄 Dữ liệu đã upload")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Số dòng hợp lệ", f"{len(clean_df):,}")

    with col_b:
        st.metric("Ngày bắt đầu", str(clean_df["time"].min().date()))

    with col_c:
        st.metric("Ngày cuối", str(clean_df["time"].max().date()))

    try:
        result_df = predict_from_dataframe(
            df_input=clean_df,
            bundle=bundle,
            device=DEVICE,
            max_rows=max_rows,
        )
    except Exception as error:
        st.error(f"Lỗi dự báo: {error}")
        st.stop()

    latest_result = result_df.iloc[-1]

    st.subheader("🎯 Dự báo phiên kế tiếp")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:
        st.metric("Model", bundle["model_name"])

    with metric_2:
        st.metric("Window", bundle["window_size"])

    with metric_3:
        st.metric(
            "Close hiện tại",
            f"{latest_result['last_close']:,.2f}",
        )

    with metric_4:
        st.metric(
            "Close dự báo",
            f"{latest_result['predicted_next_close']:,.2f}",
            f"{latest_result['predicted_change_pct']:.2f}%",
        )

    st.write(
        f"Ngày input mới nhất: **{pd.to_datetime(latest_result['input_end_date']).date()}**"
    )

    st.subheader("📊 Biểu đồ giá thực tế và giá dự báo")

    chart_df = build_chart(result_df)

    if chart_df is not None:
        st.line_chart(chart_df, width="stretch")
    else:
        st.info("Chưa đủ dữ liệu có giá thực tế t+1 để vẽ biểu đồ so sánh.")

    st.subheader("📋 Bảng kết quả dự báo")

    table_df = format_result_table(result_df, descending=True)
    st.dataframe(table_df, width="stretch", height=360)

    csv_bytes = table_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    st.download_button(
        label="⬇️ Tải kết quả dự báo CSV",
        data=csv_bytes,
        file_name="fpt_prediction_results.csv",
        mime="text/csv",
    )

    st.subheader("✍️ Nhập tay thêm 1 phiên mới")

    latest_row = clean_df.iloc[-1]

    with st.form("manual_input_form"):
        manual_col_1, manual_col_2, manual_col_3 = st.columns(3)

        with manual_col_1:
            input_date = st.date_input(
                "Ngày",
                value=pd.to_datetime(latest_row["time"]).date(),
            )

            input_open = st.number_input(
                "Open",
                value=float(latest_row["open"]),
                step=100.0,
            )

        with manual_col_2:
            input_high = st.number_input(
                "High",
                value=float(latest_row["high"]),
                step=100.0,
            )

            input_low = st.number_input(
                "Low",
                value=float(latest_row["low"]),
                step=100.0,
            )

        with manual_col_3:
            input_close = st.number_input(
                "Close",
                value=float(latest_row["close"]),
                step=100.0,
            )

            input_volume = st.number_input(
                "Volume",
                value=float(latest_row["volume"]),
                step=1000.0,
            )

        submitted = st.form_submit_button("Dự báo với dòng nhập tay")

    if submitted:
        manual_row = pd.DataFrame(
            [
                {
                    "time": pd.to_datetime(input_date),
                    "open": input_open,
                    "high": input_high,
                    "low": input_low,
                    "close": input_close,
                    "volume": input_volume,
                }
            ]
        )

        manual_df = pd.concat([clean_df, manual_row], ignore_index=True)
        manual_df = clean_stock_data(manual_df)

        manual_result_df = predict_from_dataframe(
            df_input=manual_df,
            bundle=bundle,
            device=DEVICE,
            max_rows=1,
        )

        manual_latest = manual_result_df.iloc[-1]

        st.success("Đã dự báo với dòng nhập tay.")

        result_col_1, result_col_2, result_col_3 = st.columns(3)

        with result_col_1:
            st.metric(
                "Close input",
                f"{manual_latest['last_close']:,.2f}",
            )

        with result_col_2:
            st.metric(
                "Close dự báo",
                f"{manual_latest['predicted_next_close']:,.2f}",
            )

        with result_col_3:
            st.metric(
                "% thay đổi dự báo",
                f"{manual_latest['predicted_change_pct']:.2f}%",
            )

        st.dataframe(
            format_result_table(manual_result_df, descending=True),
            width="stretch",
        )


if __name__ == "__main__":
    main()