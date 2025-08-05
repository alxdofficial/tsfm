import torch

class CorrelationSummaryProcessor:
    def __init__(self):
        self.feature_dim = 3  # [argmax_idx/D, argmin_idx/D, mean_corr]

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: Tensor of shape (B, T, D)
        Returns:
            Tensor of shape (B, D, 3)
        """
        B, T, D = patch.shape
        device = patch.device
        out = torch.zeros((B, D, self.feature_dim), dtype=torch.float32, device=device)

        for b in range(B):
            x = patch[b]  # (T, D)

            if D == 1:
                out[b, 0] = torch.tensor([0.0, 0.0, 1.0], device=device)
                continue

            # Compute correlation matrix (D, D)
            x_centered = x - x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-6
            x_norm = x_centered / std
            corr = (x_norm.T @ x_norm) / (T - 1)
            corr = torch.nan_to_num(corr, nan=0.0)

            for d in range(D):
                row = corr[d]  # (D,)
                row_no_diag = row.clone()
                row_no_diag[d] = -float('inf')
                argmax_idx = torch.argmax(row_no_diag)
                row_no_diag[d] = float('inf')
                argmin_idx = torch.argmin(row_no_diag)

                mask = torch.ones(D, dtype=torch.bool, device=device)
                mask[d] = False
                mean_corr = torch.mean(torch.abs(row[mask]))

                out[b, d] = torch.tensor([
                    argmax_idx.item() / D,
                    argmin_idx.item() / D,
                    mean_corr.item()
                ], device=device)

        return out  # (B, D, 3)
