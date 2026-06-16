from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from training.config import ExperimentConfig


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, output_size: int = 1, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.linear(out)


class CNN1DLSTMSerialModel(nn.Module):
    """
    CNN1D-LSTM bản cũ: Input -> CNN -> LSTM.
    LSTM chỉ nhìn thấy feature sau CNN.
    Giữ lại class này cho ablation nếu cần.
    """
    def __init__(
        self,
        input_size: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.2,
        use_pooling: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if use_pooling else nn.Identity()

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, T, F] -> [B, F, T]
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # [B, C, T'] -> [B, T', C]

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.linear(out)


class CNN1DLSTMModel(nn.Module):
    """
    CNN1D-LSTM residual/concat:
    raw input + CNN(raw input) -> LSTM.
    """
    def __init__(
        self,
        input_size: int,
        conv_channels: int = 16,
        kernel_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.bn = nn.BatchNorm1d(conv_channels) if use_batch_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=input_size + conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x

        cnn_x = x.permute(0, 2, 1)
        cnn_x = self.conv(cnn_x)
        cnn_x = self.bn(cnn_x)
        cnn_x = self.activation(cnn_x)
        cnn_x = self.dropout_cnn(cnn_x)
        cnn_x = cnn_x.permute(0, 2, 1)

        x = torch.cat([raw_x, cnn_x], dim=-1)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.linear(out)


class CNN2DLSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        projection_size: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        pooled_feature_size = input_size // 2 // 2
        if pooled_feature_size < 1:
            raise ValueError("input_size quá nhỏ sau hai lần pooling.")

        cnn_feature_size = conv2_channels * pooled_feature_size

        self.feature_projection = nn.Sequential(
            nn.Linear(cnn_feature_size, projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, T, F] -> [B, 1, T, F]
        x = self.cnn(x)

        batch_size = x.size(0)
        sequence_length = x.size(2)

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size, sequence_length, -1)

        x = self.feature_projection(x)

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.linear(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.size(1)
        return x + self.pe[:, :time_steps, :]


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("moving_avg_kernel nên là số lẻ để giữ đúng độ dài sequence.")

        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2

        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)

        x = torch.cat([front, x, end], dim=1)
        x = x.transpose(1, 2)
        x = self.avg(x)
        x = x.transpose(1, 2)

        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class ConvDistillLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return x


class VanillaTransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_size: int = 1,
        dropout: float = 0.2,
        max_len: int = 500,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        return self.output_layer(x)


class InformerLikeModel(nn.Module):
    """
    Informer-like model:
    - Transformer encoder blocks
    - Optional Conv1D distilling layer
    - Chưa phải ProbSparse official, nhưng giữ ý tưởng distilling.
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_size: int = 1,
        dropout: float = 0.2,
        max_len: int = 500,
        use_distill: bool = True,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.encoder_blocks = nn.ModuleList()
        self.distill_blocks = nn.ModuleList()

        for index in range(num_layers):
            block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )

            self.encoder_blocks.append(block)

            if use_distill and index < num_layers - 1:
                self.distill_blocks.append(ConvDistillLayer(d_model))

        self.use_distill = use_distill
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)

        for index, block in enumerate(self.encoder_blocks):
            x = block(x)

            if self.use_distill and index < len(self.distill_blocks):
                if x.size(1) > 2:
                    x = self.distill_blocks[index](x)

        x = x[:, -1, :]
        x = self.dropout(x)
        return self.output_layer(x)


class AutoformerLikeModel(nn.Module):
    """
    Autoformer-like model:
    - decomposition: seasonal + trend
    - Transformer encoder cho seasonal
    - trend đi nhánh riêng rồi cộng lại
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        output_size: int = 1,
        dropout: float = 0.2,
        moving_avg_kernel: int = 5,
        max_len: int = 500,
    ):
        super().__init__()

        self.decomp = SeriesDecomp(kernel_size=moving_avg_kernel)

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.seasonal_head = nn.Linear(d_model, output_size)

        self.trend_head = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)

        seasonal = self.input_projection(seasonal)
        seasonal = self.positional_encoding(seasonal)
        seasonal = self.encoder(seasonal)
        seasonal = seasonal[:, -1, :]
        seasonal = self.dropout(seasonal)

        seasonal_out = self.seasonal_head(seasonal)

        trend_last = trend[:, -1, :]
        trend_out = self.trend_head(trend_last)

        return seasonal_out + trend_out


class LastCloseResidualWrapper(nn.Module):
    """
    prediction_scaled = close_t_scaled_to_target_space + correction_scaled
    """
    def __init__(
        self,
        base_model: nn.Module,
        close_feature_index: int,
        close_feature_mean: float,
        close_feature_std: float,
        target_mean: float,
        target_std: float,
        correction_scale: float = 1.0,
    ):
        super().__init__()

        self.base_model = base_model
        self.close_feature_index = close_feature_index
        self.close_feature_mean = close_feature_mean
        self.close_feature_std = close_feature_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.correction_scale = correction_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        close_t_scaled_feature = x[:, -1, self.close_feature_index]

        close_t_raw = close_t_scaled_feature * self.close_feature_std + self.close_feature_mean
        close_t_scaled_target = (close_t_raw - self.target_mean) / self.target_std
        close_t_scaled_target = close_t_scaled_target.unsqueeze(-1)

        correction = self.base_model(x)

        return close_t_scaled_target + self.correction_scale * correction


@dataclass
class ModelFactoryContext:
    input_size: int
    close_feature_index: int
    close_feature_mean: float
    close_feature_std: float
    target_mean: float
    target_std: float
    use_last_close_anchor: bool = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def apply_last_close_anchor(base_model: nn.Module, context: ModelFactoryContext) -> nn.Module:
    if not context.use_last_close_anchor:
        return base_model

    return LastCloseResidualWrapper(
        base_model=base_model,
        close_feature_index=context.close_feature_index,
        close_feature_mean=context.close_feature_mean,
        close_feature_std=context.close_feature_std,
        target_mean=context.target_mean,
        target_std=context.target_std,
        correction_scale=1.0,
    )


def build_single_model(
    model_name: str,
    context: ModelFactoryContext,
    config: ExperimentConfig,
) -> nn.Module:
    input_size = context.input_size

    if model_name == "lstm":
        return LSTMModel(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=1,
            output_size=1,
            dropout=config.dropout_lstm,
        )

    if model_name == "cnn1d_lstm":
        base_model = CNN1DLSTMModel(
            input_size=input_size,
            conv_channels=config.cnn_conv_channels,
            kernel_size=3,
            hidden_size=64,
            num_layers=1,
            output_size=1,
            dropout=config.dropout_complex,
            use_batch_norm=True,
        )
        return apply_last_close_anchor(base_model, context)

    if model_name == "transformer":
        base_model = VanillaTransformerModel(
            input_size=input_size,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff,
            output_size=1,
            dropout=config.dropout_complex,
        )
        return apply_last_close_anchor(base_model, context)

    if model_name == "informer":
        base_model = InformerLikeModel(
            input_size=input_size,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff,
            output_size=1,
            dropout=config.dropout_complex,
            use_distill=False,
        )
        return apply_last_close_anchor(base_model, context)

    if model_name == "autoformer":
        base_model = AutoformerLikeModel(
            input_size=input_size,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff,
            output_size=1,
            dropout=config.dropout_complex,
            moving_avg_kernel=3,
        )
        return apply_last_close_anchor(base_model, context)

    if model_name == "cnn2d_lstm":
        return CNN2DLSTMModel(input_size=input_size)

    raise ValueError(f"Unknown model_name: {model_name}")
