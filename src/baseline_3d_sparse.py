from __future__ import annotations

import torch

from .model_main import BoundaryInteriorSparseRNN


class Sparse3DBaseline(BoundaryInteriorSparseRNN):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_types = torch.ones_like(self.node_types)
