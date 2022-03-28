from typing import Optional
from numpy import ndarray
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer
from aptlab_analysis.encoders import Encoder


class SequenceEncoder(Encoder):
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, values: ndarray, **kwargs) -> Tensor:
        x = self.model.encode(
            values,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
            **kwargs
        )
        return x
