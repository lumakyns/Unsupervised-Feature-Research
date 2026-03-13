import torch
import torch.nn as nn
import torch.nn.functional as F


class K_Sparse_Autoencoder(nn.Module):
    def __init__(
        self,
        dim: tuple,
        k: int,
        total_epochs: int,
        dataset_size: int,
        a: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim, self.bottleneck_dim = dim
        self.a = a
        self.k = k  # number of activations to keep per sample
        self.total_epochs = total_epochs
        self.dataset_size = dataset_size

        self.encoder      = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.identity     = nn.Identity()
        self.decoder_bias = nn.Parameter(torch.zeros(self.input_dim))

    def _apply_topk_mask(self, activations: torch.Tensor, k_count: int) -> torch.Tensor:
        k_count = min(max(1, int(k_count)), activations.shape[1])
        
        _, topk_idx = torch.topk(activations, k_count, dim=1)
        mask = torch.zeros_like(activations)
        mask.scatter_(1, topk_idx, 1)
        
        return activations * mask

    def forward(
        self,
        x: torch.Tensor,
        epoch: int = 0,
        inputs_processed_in_epoch: int = 0,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        z1 = self.encoder(x)
        a1 = self.identity(z1)

        # k annealing, so that hidden units have time to learn
        if self.training:
            current_samples = epoch * self.dataset_size + inputs_processed_in_epoch
            anneal_samples  = (self.total_epochs // 2) * self.dataset_size
            
            if anneal_samples > 0:
                progress = min(current_samples / anneal_samples, 1.0)
            else:
                progress = 1.0

            current_k = self.bottleneck_dim + progress * (self.k - self.bottleneck_dim)
        else:
            current_k = self.a * self.k

        k_count = min(max(1, int(current_k)), self.bottleneck_dim)
        a1 = self._apply_topk_mask(a1, k_count)
        z2 = F.linear(a1, self.encoder.weight.t(), self.decoder_bias)

        self.last_k = k_count
        return z2
