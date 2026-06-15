from datetime import date

import pandas as pd

from data_pipeline.base import *


SYMBOL = "FPT"


def crawl_fpt(
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> pd.DataFrame:

    q = Quote(
        symbol=SYMBOL,
        source=SOURCE
    )

    df = q.history(
        start=start_date,
        end=end_date
    )

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])

    df["symbol"] = SYMBOL
    df["source"] = SOURCE
    df["collected_date"] = str(date.today())

    df = (
        df
        .sort_values("time")
        .drop_duplicates(
            subset=["time", "symbol"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df


def main() -> None:
    df = crawl_fpt()

    output_path = RAW_DIR / "fpt_stock_price.csv"

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
        print(df.head())


if __name__ == "__main__":
    main()