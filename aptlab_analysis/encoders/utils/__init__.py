from typing import Mapping
import torch
from torch import Tensor
from pandas import DataFrame
from aptlab_analysis.encoders import Encoder


def encode_data_with_column_and_concat(
    df: DataFrame, column_encoder_mapping: Mapping[str, Encoder]
) -> Tensor:

    xs = [
        encoder(df[column].to_numpy())
        for column, encoder in column_encoder_mapping.items()
    ]
    x = torch.cat(xs, dim=-1)
    return x
