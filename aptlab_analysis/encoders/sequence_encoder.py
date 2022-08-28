from typing import Optional
from numpy import ndarray
from pandas import isna
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer
from aptlab_analysis.encoders import Encoder


class SequenceEncoder(Encoder):
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        # torch.manual_seed(123456)
        # torch.cuda.manual_seed(123456)
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, values: ndarray, **kwargs) -> Tensor:
        values = values.astype(str)
        values[isna(values)] = ""

        x = self.model.encode(
            values.tolist(), show_progress_bar=True, convert_to_tensor=True, **kwargs
        )
        assert isinstance(x, Tensor)
        return x
