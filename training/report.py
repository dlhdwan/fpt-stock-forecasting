import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]

def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT_DIR / path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Weekly Report and Charts for FPT Model.")
    parser.add_argument("--artifact-dir", default="artifacts/raw_fpt_only_residual_cnnlstm_transformer", help="Path to trained model artifact folder.")
    return parser.parse_args()

def main():
    args = parse_args()
    artifact_dir = resolve_project_path(args.artifact_dir)
    
    # Trỏ vào thư mục output
    output_dir = artifact_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "weekly_predictions.csv"
    metadata_path = output_dir / "evaluation_metadata.json"

    if not predictions_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing predictions or metadata files in {output_dir}")

    # Đọc Metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Đọc dữ liệu dự đoán
    eval_df = pd.read_csv(predictions_path)

    # SINH BIỂU ĐỒ (ACTUAL vs PREDICTED)
    eval_df_sorted = eval_df.sort_values("predict_for_date")
    dates = pd.to_datetime(eval_df_sorted["predict_for_date"])

    plt.figure(figsize=(12, 6))
    plt.plot(dates, eval_df_sorted["actual_next_close"], label='Thực tế (Actual)', color='#2ca02c', marker='o', markersize=4)
    plt.plot(dates, eval_df_sorted["predicted_next_close"], label='Dự đoán (Predicted)', color='#1f77b4', marker='x', markersize=4)

    plt.title(f"FPT Stock Price Prediction vs Actual\n({metadata['evaluation_window']['start']} to {metadata['evaluation_window']['end']})")
    plt.xlabel("Ngày")
    plt.ylabel("Giá đóng cửa (Close Price)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Lưu biểu đồ vào output_dir
    chart_path = output_dir / "predictions_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # SINH FILE REPORT / DASHBOARD (MARKDOWN)
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Dashboard Đánh Giá Mô Hình FPT\n\n")
        f.write(f"Thời gian đánh giá: {metadata['evaluation_time']} (Giờ VN)\n\n")
        f.write(f"Model Checkpoint: `{metadata['model_info']['checkpoint']}`\n\n")
        f.write(f"Giai đoạn đánh giá: Từ `{metadata['evaluation_window']['start']}` đến `{metadata['evaluation_window']['end']}` ({metadata['evaluation_window']['rows']} ngày)\n\n")
        
        f.write("## 📈 Tổng quan Metrics\n\n")
        f.write("| Metric | Giá trị |\n")
        f.write("|---|---|\n")
        for k, v in metadata['metrics'].items():
            f.write(f"| **{k.upper()}** | `{v:.4f}` |\n")
            
        f.write("\nBiểu đồ Thực tế vs Dự đoán\n\n")
        f.write("![Predictions Chart](./predictions_chart.png)\n")

    print(f"Đã tạo thành công Chart và Report tại: {output_dir}")

if __name__ == "__main__":
    main()