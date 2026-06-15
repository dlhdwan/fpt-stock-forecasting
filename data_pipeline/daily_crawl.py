from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from data_pipeline.base import RAW_DIR, START_DATE
from data_pipeline.build_dataset import build_dataset

from data_pipeline.crawl_fpt import crawl_fpt
from data_pipeline.crawl_vnindex import crawl_vnindex



PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def get_next_start_date(
    csv_path: Path
) -> str:

    if not csv_path.exists():
        return START_DATE

    df = pd.read_csv(csv_path)

    if df.empty:
        return START_DATE

    df["time"] = pd.to_datetime(
        df["time"]
    )

    latest_date = df["time"].max()

    next_date = latest_date + timedelta(
        days=1
    )

    return next_date.strftime(
        "%Y-%m-%d"
    )


def merge_old_new_data(
    csv_path: Path,
    new_df: pd.DataFrame,
    subset_cols=None
) -> pd.DataFrame:

    if subset_cols is None:
        subset_cols = ["time", "symbol"]

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
    else:
        old_df = pd.DataFrame()

    if old_df.empty and new_df.empty:
        return pd.DataFrame()

    if old_df.empty:
        merged_df = new_df.copy()
    elif new_df.empty:
        merged_df = old_df.copy()
    else:
        merged_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

    merged_df["time"] = pd.to_datetime(
        merged_df["time"]
    )

    merged_df = (
        merged_df
        .sort_values("time")
        .drop_duplicates(
            subset=subset_cols,
            keep="last"
        )
        .reset_index(drop=True)
    )

    merged_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    return merged_df


def crawl_and_update(
    name: str,
    csv_file: str,
    crawl_func
) -> None:

    csv_path = RAW_DIR / csv_file

    start_date = get_next_start_date(
        csv_path
    )

    end_date = str(date.today())

    print("=" * 80)
    print(f"Crawling {name}")
    print(f"Range: {start_date} -> {end_date}")

    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print(f"{name}: no missing days.")
        return

    new_df = crawl_func(
        start_date=start_date,
        end_date=end_date
    )

    print(f"New rows: {new_df.shape}")

    merged_df = merge_old_new_data(
        csv_path=csv_path,
        new_df=new_df
    )

    print(f"Saved: {csv_path}")
    print(f"Total rows: {merged_df.shape}")

    if not merged_df.empty:
        print("Min date:", merged_df["time"].min())
        print("Max date:", merged_df["time"].max())


def rebuild_dataset() -> None:

    print("=" * 80)
    print("Rebuilding processed dataset...")

    df = build_dataset()

    output_path = (
        PROCESSED_DIR /
        "merged_dataset.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Saved:", output_path)
    print("Shape:", df.shape)

    if not df.empty:
        print("Min date:", df["time"].min())
        print("Max date:", df["time"].max())


def daily_crawl() -> None:

    crawl_and_update(
        name="FPT stock price",
        csv_file="fpt_stock_price.csv",
        crawl_func=crawl_fpt
    )

    crawl_and_update(
        name="VNINDEX price",
        csv_file="vnindex_price.csv",
        crawl_func=crawl_vnindex
    )


    rebuild_dataset()


if __name__ == "__main__":
    daily_crawl()