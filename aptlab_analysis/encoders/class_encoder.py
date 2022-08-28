from typing import Dict
from numpy import ndarray
import torch
from torch import Tensor
from aptlab_analysis.encoders import Encoder


class ClassEncoder(Encoder):
    def __init__(self, mapping: Dict[str, int], has_None_class: bool = False):
        self.mapping = mapping
        self.has_None_class = has_None_class

    def __call__(self, values: ndarray) -> Tensor:
        offset = 0
        class_count = len(self.mapping)

        x = torch.zeros(len(values), class_count, dtype=torch.long)
        for i, column in enumerate(values.astype(str)):
            if column in self.mapping:
                x[i, self.mapping[column] + offset] = 1
            elif column == "nan":
                if self.has_None_class:
                    x[i, 0] = 1
            else:
                raise KeyError(column)
        return x
