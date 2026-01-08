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

    def __init__(self, queue_size: int = 4096, embedding_dim: int = 256):
        """
        Initialize memory bank.

        Args:
            queue_size: Number of past embeddings to store (default 4096)
            embedding_dim: Dimension of embeddings (default 256)
        """
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim

        # Initialize queues
        self.imu_queue = torch.zeros(queue_size, embedding_dim)
        self.text_queue = torch.zeros(queue_size, embedding_dim)  # Cache text embeddings
        self.ptr = 0  # Pointer to next position to fill
        self.is_full = False  # Whether queue has been filled once

    def update(self, imu_emb: torch.Tensor, text_emb: torch.Tensor):
        """
        Add new embeddings to queue (FIFO - First In First Out).

        Args:
            imu_emb: IMU embeddings to add (batch_size, embedding_dim)
            text_emb: Text embeddings to add (batch_size, embedding_dim)
        """
        batch_size = imu_emb.shape[0]

        # Move embeddings to CPU to save GPU memory
        imu_emb_cpu = imu_emb.detach().cpu()
        text_emb_cpu = text_emb.detach().cpu()

        # Calculate end pointer
        end_ptr = self.ptr + batch_size

        if end_ptr <= self.queue_size:
            # Simple case: no wraparound
            self.imu_queue[self.ptr:end_ptr] = imu_emb_cpu
            self.text_queue[self.ptr:end_ptr] = text_emb_cpu
        else:
            # Wraparound case: split across boundary
            first_part_size = self.queue_size - self.ptr

            # Fill to end of queue
            self.imu_queue[self.ptr:] = imu_emb_cpu[:first_part_size]
            self.text_queue[self.ptr:] = text_emb_cpu[:first_part_size]

            # Wrap around to beginning
            remainder = batch_size - first_part_size
            self.imu_queue[:remainder] = imu_emb_cpu[first_part_size:]
            self.text_queue[:remainder] = text_emb_cpu[first_part_size:]

            self.is_full = True

        # Update pointer (circular)
        self.ptr = end_ptr % self.queue_size

    def get_queue_embeddings(self, device: torch.device):
        """
        Get queue embeddings for loss computation.

        Returns cached text embeddings (no re-encoding needed).
        Text embeddings are cached when update() is called.

        Args:
            device: Device to move embeddings to

        Returns:
            Tuple of (imu_queue, text_queue) on specified device
        """
        # Get actual queue size (might be less than queue_size if not full yet)
        actual_size = self.queue_size if self.is_full else self.ptr

        if actual_size == 0:
            # Queue is empty - return empty tensors
            return (
                torch.zeros(0, self.embedding_dim, device=device),
                torch.zeros(0, self.embedding_dim, device=device)
            )

        # Get active portion of queue (already cached)
        active_imu = self.imu_queue[:actual_size]
        active_text = self.text_queue[:actual_size]

        # Move embeddings to device
        imu_queue_device = active_imu.to(device)
        text_queue_device = active_text.to(device)

        return imu_queue_device, text_queue_device

    def __len__(self) -> int:
        """Return current size of queue."""
        return self.queue_size if self.is_full else self.ptr

    def __repr__(self) -> str:
        """String representation."""
        return f"MemoryBank(size={len(self)}/{self.queue_size}, dim={self.embedding_dim})"

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'imu_queue': self.imu_queue.clone(),
            'text_queue': self.text_queue.clone(),
            'ptr': self.ptr,
            'is_full': self.is_full,
            'queue_size': self.queue_size,
            'embedding_dim': self.embedding_dim
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.imu_queue = state_dict['imu_queue']
        self.text_queue = state_dict['text_queue']
        self.ptr = state_dict['ptr']
        self.is_full = state_dict['is_full']
        # Verify dimensions match
        if state_dict.get('queue_size') != self.queue_size:
            print(f"  Warning: Queue size mismatch (checkpoint={state_dict.get('queue_size')}, current={self.queue_size})")
        if state_dict.get('embedding_dim') != self.embedding_dim:
            print(f"  Warning: Embedding dim mismatch (checkpoint={state_dict.get('embedding_dim')}, current={self.embedding_dim})")
