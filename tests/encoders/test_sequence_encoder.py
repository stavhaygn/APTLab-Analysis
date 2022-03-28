import numpy as np
import torch
from aptlab_analysis.encoders import SequenceEncoder


def test_sequence_encoder():
    values = np.array(
        ["c:\windows\system32\windowspowershell\v1.0\powershell.exe"], dtype=object
    )
    sequence_encoder = SequenceEncoder()
    x = sequence_encoder(values)
    assert x.shape == torch.Size([1, 384])
