"""
Memory Bank for MoCo-style contrastive learning.

Maintains a queue of past embeddings to use as additional negatives
in contrastive loss, enabling large effective batch sizes with limited GPU memory.

Note: Queue items are treated as hard negatives (standard MoCo practice).
Text labels are NOT stored - computing soft targets for queue negatives would
require re-encoding 512+ labels every batch, which is computationally expensive
and not supported by literature (MoCo, CLIP use hard negatives for queues).
"""

import torch


class MemoryBank:
    """
    MoCo-style memory bank for storing past embeddings as negatives.

    Maintains a FIFO queue of (IMU embeddings, text embeddings) from previous batches
    to provide many more negative samples for InfoNCE contrastive loss.

    Example:
        Batch size: 16
        Queue size: 4096
        → Each sample compared against 15 + 4096 = 4111 negatives!
    """

    def __init__(self, queue_size: int = 4096, embedding_dim: int = 256, device: torch.device = None):
        """
        Initialize memory bank.

        Args:
            queue_size: Number of past embeddings to store (default 4096)
            embedding_dim: Dimension of embeddings (default 256)
            device: Device to store queue on (default: CPU). Set to cuda device
                    to avoid CPU↔GPU transfers every step (~750KB for 256×384).
        """
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cpu')

        # Initialize queues on specified device
        self.imu_queue = torch.zeros(queue_size, embedding_dim, device=self.device)
        self.text_queue = torch.zeros(queue_size, embedding_dim, device=self.device)
        self.ptr = 0  # Pointer to next position to fill
        self.is_full = False  # Whether queue has been filled once

    def update(self, imu_emb: torch.Tensor, text_emb: torch.Tensor):
        """
        Add new embeddings to queue (FIFO - First In First Out).

        Embeddings are stored on self.device (no CPU round-trips if device=cuda).

        Args:
            imu_emb: IMU embeddings to add (batch_size, embedding_dim)
            text_emb: Text embeddings to add (batch_size, embedding_dim)
        """
        # Detach and move to queue device (no-op if already on same device)
        imu_detached = imu_emb.detach().to(self.device)
        text_detached = text_emb.detach().to(self.device)

        # Validate embeddings - NaN or zero-norm indicate bugs that should be fixed
        imu_nan_mask = torch.isnan(imu_detached).any(dim=1)
        text_nan_mask = torch.isnan(text_detached).any(dim=1)
        if imu_nan_mask.any() or text_nan_mask.any():
            nan_indices = (imu_nan_mask | text_nan_mask).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"NaN detected in embeddings at indices {nan_indices}. "
                f"This indicates a numerical bug in the model forward pass."
            )

        imu_zero_mask = imu_detached.norm(dim=1) < 1e-6
        text_zero_mask = text_detached.norm(dim=1) < 1e-6
        if imu_zero_mask.any() or text_zero_mask.any():
            zero_indices = (imu_zero_mask | text_zero_mask).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"Zero-norm embeddings detected at indices {zero_indices}. "
                f"This indicates invalid samples that should have been filtered during data loading."
            )

        batch_size = imu_detached.shape[0]

        # If batch is larger than queue, only keep the last queue_size embeddings
        if batch_size >= self.queue_size:
            self.imu_queue[:] = imu_detached[-self.queue_size:]
            self.text_queue[:] = text_detached[-self.queue_size:]
            self.ptr = 0
            self.is_full = True
            return

        # Calculate end pointer
        end_ptr = self.ptr + batch_size

        if end_ptr <= self.queue_size:
            # Simple case: no wraparound
            self.imu_queue[self.ptr:end_ptr] = imu_detached
            self.text_queue[self.ptr:end_ptr] = text_detached
            if end_ptr == self.queue_size:
                self.is_full = True
        else:
            # Wraparound case: split across boundary
            first_part_size = self.queue_size - self.ptr

            # Fill to end of queue
            self.imu_queue[self.ptr:] = imu_detached[:first_part_size]
            self.text_queue[self.ptr:] = text_detached[:first_part_size]

            # Wrap around to beginning
            remainder = batch_size - first_part_size
            self.imu_queue[:remainder] = imu_detached[first_part_size:]
            self.text_queue[:remainder] = text_detached[first_part_size:]

            self.is_full = True

        # Update pointer (circular)
        self.ptr = end_ptr % self.queue_size

    def get_queue_embeddings(self, device: torch.device = None):
        """
        Get queue embeddings for loss computation.

        Returns cached text embeddings (no re-encoding needed).
        When queue is on GPU, this is a zero-copy slice (no transfer).

        Args:
            device: Device to move embeddings to (default: self.device)

        Returns:
            Tuple of (imu_queue, text_queue) on specified device
        """
        if device is None:
            device = self.device

        # Get actual queue size (might be less than queue_size if not full yet)
        actual_size = self.queue_size if self.is_full else self.ptr

        if actual_size == 0:
            # Queue is empty - return empty tensors
            return (
                torch.zeros(0, self.embedding_dim, device=device),
                torch.zeros(0, self.embedding_dim, device=device)
            )

        # Get active portion of queue
        # No-op if already on target device (typical case with GPU queue)
        active_imu = self.imu_queue[:actual_size].to(device)
        active_text = self.text_queue[:actual_size].to(device)

        return active_imu, active_text

    def __len__(self) -> int:
        """Return current size of queue."""
        return self.queue_size if self.is_full else self.ptr

    def __repr__(self) -> str:
        """String representation."""
        return f"MemoryBank(size={len(self)}/{self.queue_size}, dim={self.embedding_dim})"

    def state_dict(self) -> dict:
        """Return state for checkpointing (always saved on CPU)."""
        return {
            'imu_queue': self.imu_queue.cpu().clone(),
            'text_queue': self.text_queue.cpu().clone(),
            'ptr': self.ptr,
            'is_full': self.is_full,
            'queue_size': self.queue_size,
            'embedding_dim': self.embedding_dim
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint (moves to self.device)."""
        self.imu_queue = state_dict['imu_queue'].to(self.device)
        self.text_queue = state_dict['text_queue'].to(self.device)
        self.ptr = state_dict['ptr']
        self.is_full = state_dict['is_full']
        # Verify dimensions match
        if state_dict.get('queue_size') != self.queue_size:
            print(f"  Warning: Queue size mismatch (checkpoint={state_dict.get('queue_size')}, current={self.queue_size})")
        if state_dict.get('embedding_dim') != self.embedding_dim:
            print(f"  Warning: Embedding dim mismatch (checkpoint={state_dict.get('embedding_dim')}, current={self.embedding_dim})")
