from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.base import MIN_DATE


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


OUTPUT_FEATURE_COLUMNS = [
    # Core FPT OHLCV
    "open",
    "high",
    "low",
    "close",
    "volume",

    # VNIndex: only close price + features derived from close
    "vnindex_close",

    # FPT returns
    "return_1d",
    "log_return_1d",

    # Moving average / trend features
    "ma5",
    "ma10",
    "ma20",
    "close_ma5_ratio",
    "close_ma20_ratio",
    "ma5_ma20_ratio",

    # Momentum indicators
    "momentum_5d",
    "momentum_10d",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",

    # Volume features
    "volume_ratio_5",
    "volume_ratio_20",

    # Basic price volatility features
    "volatility_5d",
    "volatility_20d",
    "high_low_pct",
    "open_close_pct",
    "atr14",
    "bollinger_width20",

    # VNIndex close-derived features
    "vnindex_return_1d",
    "vnindex_ma20",
    "vnindex_close_ma20_ratio",
    "vnindex_volatility_20d",
]

TARGET_COL = "target_next_close"


def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def load_csv(file_name: str) -> pd.DataFrame:
    path = RAW_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df[df["time"] >= pd.to_datetime(MIN_DATE)]

    df = (
        df
        .dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    return df


def prepare_fpt_price() -> pd.DataFrame:
    df = load_csv("fpt_stock_price.csv")

    keep_cols = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in FPT data: {missing_cols}")

    return df[keep_cols]


def prepare_vnindex_close() -> pd.DataFrame:
    df = load_csv("vnindex_price.csv")

    keep_cols = [
        "time",
        "close",
    ]

    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in VNIndex data: {missing_cols}")

    df = df[keep_cols].rename(columns={"close": "vnindex_close"})

    return df


def add_fpt_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Return features
    df["return_1d"] = df["close"].pct_change()

    close_ratio = safe_divide(df["close"], df["close"].shift(1))
    close_ratio = close_ratio.where(close_ratio > 0)
    df["log_return_1d"] = np.log(close_ratio)

    # Moving averages
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["ma20"] = df["close"].rolling(window=20).mean()

    # Relative trend features to reduce pure price-level redundancy
    df["close_ma5_ratio"] = safe_divide(df["close"], df["ma5"]) - 1
    df["close_ma20_ratio"] = safe_divide(df["close"], df["ma20"]) - 1
    df["ma5_ma20_ratio"] = safe_divide(df["ma5"], df["ma20"]) - 1

    # Momentum indicators
    df["momentum_5d"] = df["close"].pct_change(periods=5)
    df["momentum_10d"] = df["close"].pct_change(periods=10)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volume features
    volume_ma5 = df["volume"].rolling(window=5).mean()
    volume_ma20 = df["volume"].rolling(window=20).mean()

    df["volume_ratio_5"] = safe_divide(df["volume"], volume_ma5) - 1
    df["volume_ratio_20"] = safe_divide(df["volume"], volume_ma20) - 1

    # Basic price volatility features
    df["volatility_5d"] = df["return_1d"].rolling(window=5).std()
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std()

    df["high_low_pct"] = safe_divide(df["high"] - df["low"], df["close"])
    df["open_close_pct"] = safe_divide(df["close"] - df["open"], df["open"])

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr14"] = true_range.rolling(window=14).mean()

    ma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()

    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20

    df["bollinger_width20"] = safe_divide(upper_band - lower_band, ma20)

    return df


def add_vnindex_close_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["vnindex_return_1d"] = df["vnindex_close"].pct_change()
    df["vnindex_ma20"] = df["vnindex_close"].rolling(window=20).mean()

    df["vnindex_close_ma20_ratio"] = (
        safe_divide(df["vnindex_close"], df["vnindex_ma20"]) - 1
    )

    df["vnindex_volatility_20d"] = (
        df["vnindex_return_1d"]
        .rolling(window=20)
        .std()
    )

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-step-ahead close price prediction.
    # No target_next_return is created in this cleaned dataset.
    df[TARGET_COL] = df["close"].shift(-1)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    output_cols = ["time"] + OUTPUT_FEATURE_COLUMNS + [TARGET_COL]

    missing_cols = [col for col in output_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing output columns: {missing_cols}")

    df = df[output_cols]

    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows created by rolling indicators and the final row without next-day target.
    df = df.dropna().reset_index(drop=True)

    df = df[df["time"] >= pd.to_datetime(MIN_DATE)]
    df = df.sort_values("time").reset_index(drop=True)

    return df


def build_dataset() -> pd.DataFrame:
    fpt = prepare_fpt_price()
    vnindex = prepare_vnindex_close()

    print("FPT:", fpt.shape, fpt["time"].min(), fpt["time"].max())
    print("VNINDEX close:", vnindex.shape, vnindex["time"].min(), vnindex["time"].max())

    df = fpt.merge(vnindex, on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)

    df = add_fpt_price_features(df)
    df = add_vnindex_close_features(df)
    df = add_target(df)
    df = clean_dataset(df)

    return df


def main() -> None:
    df = build_dataset()

    output_path = PROCESSED_DIR / "merged_dataset.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("Saved:", output_path)
    print("Shape:", df.shape)
    print("Min date:", df["time"].min())
    print("Max date:", df["time"].max())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\nHead:")
    print(df.head())

    print("\nTail:")
    print(df.tail())


if __name__ == "__main__":
    main()
