"""
Memory Bank for MoCo-style contrastive learning.

Maintains a queue of past embeddings to use as additional negatives
in contrastive loss, enabling large effective batch sizes with limited GPU memory.

By default, queue items are treated as hard negatives (standard MoCo practice).

EXP-P1 (queue ablation): when constructed with `frozen_dim`, the bank ALSO stores the
frozen SBERT mean-pool text embedding for each queued entry. This lets the loss compute
soft targets over the queue (by label-text similarity) so semantically identical entries
("walking" vs "strolling") are NOT pushed apart — addressing reviewer A2. The frozen text
is already computed each step for the in-batch soft targets, so caching it in the queue is
cheap (no re-encoding): we keep the (queue_size x frozen_dim) tensor in lock-step with the
imu/text queues.
"""

import torch


class MemoryBank:
    """
    MoCo-style memory bank for storing past embeddings as negatives.

    Maintains a FIFO queue of (IMU embeddings, text embeddings) from previous batches
    to provide many more negative samples for InfoNCE contrastive loss. Optionally also
    stores frozen SBERT text embeddings (EXP-P1 semantic queue).

    Example:
        Batch size: 16
        Queue size: 4096
        → Each sample compared against 15 + 4096 = 4111 negatives!
    """

    def __init__(self, queue_size: int = 4096, embedding_dim: int = 256, device: torch.device = None,
                 frozen_dim: int = None):
        """
        Initialize memory bank.

        Args:
            queue_size: Number of past embeddings to store (default 4096)
            embedding_dim: Dimension of embeddings (default 256)
            device: Device to store queue on (default: CPU). Set to cuda device
                    to avoid CPU↔GPU transfers every step (~750KB for 256×384).
            frozen_dim: If set (EXP-P1 semantic queue), also store the frozen SBERT
                    mean-pool text embedding per entry so queued synonyms can receive
                    SOFT targets instead of being treated as hard negatives. None = the
                    standard hard-negative queue (no frozen storage).
        """
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.frozen_dim = frozen_dim
        self.device = device or torch.device('cpu')

        # Initialize queues on specified device
        self.imu_queue = torch.zeros(queue_size, embedding_dim, device=self.device)
        self.text_queue = torch.zeros(queue_size, embedding_dim, device=self.device)
        # EXP-P1 semantic queue: cached frozen SBERT text embeddings for queue soft targets.
        self.frozen_queue = (torch.zeros(queue_size, frozen_dim, device=self.device)
                             if frozen_dim is not None else None)
        self.ptr = 0  # Pointer to next position to fill
        self.is_full = False  # Whether queue has been filled once

    def update(self, imu_emb: torch.Tensor, text_emb: torch.Tensor, frozen_emb: torch.Tensor = None):
        """
        Add new embeddings to queue (FIFO - First In First Out).

        Embeddings are stored on self.device (no CPU round-trips if device=cuda).

        Args:
            imu_emb: IMU embeddings to add (batch_size, embedding_dim)
            text_emb: Text embeddings to add (batch_size, embedding_dim)
            frozen_emb: (EXP-P1 semantic queue) frozen SBERT text embeddings
                (batch_size, frozen_dim). Stored only if the bank was built with frozen_dim;
                kept in lock-step with the imu/text queues (same FIFO indices).
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

        # Frozen embeddings (semantic queue) — only stored if this bank tracks them.
        frozen_detached = None
        if self.frozen_queue is not None and frozen_emb is not None:
            frozen_detached = frozen_emb.detach().to(self.device)

        batch_size = imu_detached.shape[0]

        # If batch is larger than queue, only keep the last queue_size embeddings
        if batch_size >= self.queue_size:
            self.imu_queue[:] = imu_detached[-self.queue_size:]
            self.text_queue[:] = text_detached[-self.queue_size:]
            if frozen_detached is not None:
                self.frozen_queue[:] = frozen_detached[-self.queue_size:]
            self.ptr = 0
            self.is_full = True
            return

        end_ptr = self.ptr + batch_size

        # FIFO write with circular wraparound — IDENTICAL indexing for every queue so the
        # imu/text/frozen entries at any slot always belong to the same original sample.
        def _write(queue, data):
            if end_ptr <= self.queue_size:
                queue[self.ptr:end_ptr] = data
            else:
                first = self.queue_size - self.ptr
                queue[self.ptr:] = data[:first]
                queue[:batch_size - first] = data[first:]

        _write(self.imu_queue, imu_detached)
        _write(self.text_queue, text_detached)
        if frozen_detached is not None:
            _write(self.frozen_queue, frozen_detached)

        if end_ptr >= self.queue_size:
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

    def get_frozen_queue(self, device: torch.device = None):
        """Get the frozen SBERT text embeddings for the active queue (EXP-P1 semantic mode).

        Returns the (actual_size, frozen_dim) frozen embeddings aligned 1:1 with
        get_queue_embeddings(), or None if this bank stores no frozen embeddings.
        """
        if self.frozen_queue is None:
            return None
        if device is None:
            device = self.device
        actual_size = self.queue_size if self.is_full else self.ptr
        if actual_size == 0:
            return torch.zeros(0, self.frozen_dim, device=device)
        return self.frozen_queue[:actual_size].to(device)

    def __len__(self) -> int:
        """Return current size of queue."""
        return self.queue_size if self.is_full else self.ptr

    def __repr__(self) -> str:
        """String representation."""
        mode = f", frozen_dim={self.frozen_dim}" if self.frozen_queue is not None else ""
        return f"MemoryBank(size={len(self)}/{self.queue_size}, dim={self.embedding_dim}{mode})"

    def state_dict(self) -> dict:
        """Return state for checkpointing (always saved on CPU)."""
        sd = {
            'imu_queue': self.imu_queue.cpu().clone(),
            'text_queue': self.text_queue.cpu().clone(),
            'ptr': self.ptr,
            'is_full': self.is_full,
            'queue_size': self.queue_size,
            'embedding_dim': self.embedding_dim,
            'frozen_dim': self.frozen_dim,
        }
        if self.frozen_queue is not None:
            sd['frozen_queue'] = self.frozen_queue.cpu().clone()
        return sd

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint (moves to self.device)."""
        self.imu_queue = state_dict['imu_queue'].to(self.device)
        self.text_queue = state_dict['text_queue'].to(self.device)
        self.ptr = state_dict['ptr']
        self.is_full = state_dict['is_full']
        if self.frozen_queue is not None and 'frozen_queue' in state_dict:
            self.frozen_queue = state_dict['frozen_queue'].to(self.device)
        # Verify dimensions match
        if state_dict.get('queue_size') != self.queue_size:
            print(f"  Warning: Queue size mismatch (checkpoint={state_dict.get('queue_size')}, current={self.queue_size})")
        if state_dict.get('embedding_dim') != self.embedding_dim:
            print(f"  Warning: Embedding dim mismatch (checkpoint={state_dict.get('embedding_dim')}, current={self.embedding_dim})")
