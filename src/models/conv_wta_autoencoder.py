import torch
import torch.nn as nn
import torch.nn.functional as F


class CONV_WTA_Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k: int,
    ) -> None:
        super().__init__()
        pass


    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
    ) -> torch.Tensor:
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass
