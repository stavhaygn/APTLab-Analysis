from numpy import ndarray
from torch import Tensor


class Encoder(object):
    def __call__(self, values: ndarray, **kwargs) -> Tensor:
        pass
