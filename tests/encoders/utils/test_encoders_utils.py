import pandas as pd
import torch
from aptlab_analysis.encoders import ClassEncoder
from aptlab_analysis.encoders.utils import encode_data_with_column_and_concat


def test_encode_data_with_column_and_concat():
    data = {
        "file_name_original": ["git.exe", "netsh.exe"],
        "intergrity_level": ["Medium", "High"],
    }
    df = pd.DataFrame(data=data)

    file_name_original_mapping = {"git.exe": 0, "netsh.exe": 1}
    integrity_level_mapping = {"Low": 0, "Medium": 1, "High": 2, "System": 3}

    column_encoder_mapping = {
        "file_name_original": ClassEncoder(file_name_original_mapping),
        "intergrity_level": ClassEncoder(integrity_level_mapping),
    }
    x = encode_data_with_column_and_concat(df, column_encoder_mapping)
    assert x.equal(torch.tensor([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]]))
