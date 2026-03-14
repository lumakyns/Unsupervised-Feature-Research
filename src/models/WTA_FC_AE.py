import torch
import torch.nn as nn
import torch.nn.functional as F


class WTA_FC_AE(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k: int,
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim
        self.k = k  # number of winners (activations to keep) per hidden unit

        self.encoder = nn.Linear(self.input_dim, self.bottleneck_dim, bias=False)
        self.relu    = nn.ReLU()
        self.decoder = nn.Linear(self.bottleneck_dim, self.input_dim, bias=False)

    def _apply_lifetime_sparsity(self, activations: torch.Tensor) -> torch.Tensor:
        k_count = min(max(1, self.k), activations.shape[0])
        _, topk_idx = torch.topk(activations, k_count, dim=0)
        mask = torch.zeros_like(activations)
        mask.scatter_(0, topk_idx, 1)
        return activations * mask

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        a1 = self.relu(z1)

        if self.training:
            a1 = self._apply_lifetime_sparsity(a1)
        else:
            self.last_latent = a1.detach()

        z2 = self.decoder(a1)

        return z2
