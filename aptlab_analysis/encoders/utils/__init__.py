from typing import Dict
import torch
from torch import Tensor
from pandas import DataFrame
from aptlab_analysis.encoders import Encoder


def encode_data_with_column_and_concat(
    df: DataFrame, column_encoder_mapping: Dict[str, Encoder]
) -> Tensor:

    xs = [
        encoder(df[column].values) for column, encoder in column_encoder_mapping.items()
    ]
    x = torch.cat(xs, dim=-1)
    return x
