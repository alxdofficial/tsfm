import torch
import torch.nn as nn
import numpy as np
import concurrent.futures
import math
from typing import List, Dict


class SinusoidalEncoding:
    @staticmethod
    def encode(x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (...,)
            dim: Encoding dimension (must be even)
        Returns:
            Tensor of shape (..., dim)
        """
        assert dim % 2 == 0, "Encoding dim must be even"
        device = x.device
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / dim)
        )  # (dim/2,)
        x = x.unsqueeze(-1)  # (..., 1)
        sinusoid = torch.cat([torch.sin(x * div_term), torch.cos(x * div_term)], dim=-1)  # (..., dim)
        return sinusoid