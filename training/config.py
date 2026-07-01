from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


RAW_FPT_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@dataclass
class ExperimentConfig:
    data_path: Optional[Path] = None
    artifact_dir: Optional[Path] = None

    target_col: str = "target_next_close"
    target_date_col: str = "target_date"

    window_sizes: List[int] = field(default_factory=lambda: [3, 5, 7, 30])
    model_names: List[str] = field(default_factory=lambda: [
        "lstm",
        "cnn1d_lstm",
        "transformer",
        "informer",
        "autoformer",
    ])

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    start_date: str = "2019-01-01"

    batch_size: int = 32
    epochs: int = 150
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42

    use_last_close_anchor: bool = True

    # Model hyperparameters
    lstm_hidden_size: int = 64
    cnn_conv_channels: int = 16
    transformer_d_model: int = 32
    transformer_nhead: int = 4
    transformer_layers: int = 1
    transformer_ff: int = 64
    dropout_lstm: float = 0.2
    dropout_complex: float = 0.1

    def to_jsonable(self) -> dict:
        data = asdict(self)
        if self.data_path is not None:
            data["data_path"] = str(self.data_path)
        if self.artifact_dir is not None:
            data["artifact_dir"] = str(self.artifact_dir)
        return data


def get_default_data_path() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "fpt_stock_price.csv"


def get_default_artifact_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "raw_fpt_only_residual_cnnlstm_transformer"


def build_config(
    data_path: Optional[str | Path] = None,
    artifact_dir: Optional[str | Path] = None,
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.data_path = Path(data_path) if data_path else get_default_data_path()
    cfg.artifact_dir = Path(artifact_dir) if artifact_dir else get_default_artifact_dir()
    return cfg