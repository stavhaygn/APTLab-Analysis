import numpy as np
import torch
from aptlab_analysis.encoders import ClassEncoder


def test_class_encoder():
    mapping = {"T1016": 0, "T1033": 1, "T1047": 2, "T1053": 3}
    values = np.array(
        ["T1047", "T1016", "T1033", "T1033", "T1053", "nan"], dtype=object
    )

    class_encoder = ClassEncoder(mapping)
    x = class_encoder(values)
    assert x.equal(
        torch.tensor(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    )


def test_class_encoder_has_none_class():
    mapping = {"T1016": 0, "T1033": 1, "T1047": 2, "T1053": 3}
    values = np.array(
        ["T1047", "T1016", "T1033", "T1033", "T1053", "nan"], dtype=object
    )

    class_encoder = ClassEncoder(mapping, has_None_class=True)
    x = class_encoder(values)
    assert x.equal(
        torch.tensor(
            [
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    )
