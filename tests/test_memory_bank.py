"""Test 5: Memory bank boundary conditions.

Verifies queue fill, wraparound, overflow, and dequeue behavior.
"""

import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_scripts.human_activity_recognition.memory_bank import MemoryBank


class TestMemoryBankFill:
    """Test queue filling behavior."""

    def test_empty_queue(self):
        bank = MemoryBank(queue_size=16, embedding_dim=8)
        assert len(bank) == 0
        assert not bank.is_full

    def test_single_update_size(self):
        bank = MemoryBank(queue_size=16, embedding_dim=8)
        imu = torch.randn(4, 8)
        text = torch.randn(4, 8)
        bank.update(imu, text)
        assert len(bank) == 4
        assert not bank.is_full

    def test_fills_exactly_at_queue_size(self):
        queue_size = 8
        bank = MemoryBank(queue_size=queue_size, embedding_dim=4)
        # Push exactly queue_size items
        imu = torch.randn(queue_size, 4)
        text = torch.randn(queue_size, 4)
        bank.update(imu, text)
        assert len(bank) == queue_size
        assert bank.is_full

    def test_fills_via_multiple_updates(self):
        queue_size = 16
        bank = MemoryBank(queue_size=queue_size, embedding_dim=4)
        for _ in range(4):
            imu = torch.randn(4, 4)
            text = torch.randn(4, 4)
            bank.update(imu, text)
        assert len(bank) == queue_size
        assert bank.is_full


class TestMemoryBankWraparound:
    """Test wraparound when queue is full."""

    def test_wraparound_overwrites_oldest(self):
        queue_size = 8
        bank = MemoryBank(queue_size=queue_size, embedding_dim=4)

        # Fill queue with distinguishable embeddings
        first_batch = torch.ones(queue_size, 4) * 1.0
        bank.update(first_batch, first_batch)
        assert bank.is_full

        # Push new batch -> should overwrite oldest
        new_batch = torch.ones(4, 4) * 2.0
        bank.update(new_batch, new_batch)

        imu_q, text_q = bank.get_queue_embeddings()
        # First 4 should be overwritten with 2.0, last 4 should still be 1.0
        assert torch.allclose(imu_q[:4], new_batch)
        assert torch.allclose(imu_q[4:], first_batch[4:])

    def test_size_stays_at_queue_size_after_overflow(self):
        queue_size = 8
        bank = MemoryBank(queue_size=queue_size, embedding_dim=4)
        # Push much more than queue_size
        for _ in range(10):
            bank.update(torch.randn(4, 4), torch.randn(4, 4))
        assert len(bank) == queue_size


class TestMemoryBankOverflow:
    """Test batch larger than queue_size."""

    def test_batch_larger_than_queue(self):
        queue_size = 4
        bank = MemoryBank(queue_size=queue_size, embedding_dim=8)
        # Push batch of size 10 (larger than queue_size=4)
        big_batch_imu = torch.randn(10, 8)
        big_batch_text = torch.randn(10, 8)
        bank.update(big_batch_imu, big_batch_text)

        assert bank.is_full
        assert len(bank) == queue_size

        imu_q, text_q = bank.get_queue_embeddings()
        assert imu_q.shape == (queue_size, 8)
        # Should keep the LAST queue_size items from the batch
        assert torch.allclose(imu_q, big_batch_imu[-queue_size:])


class TestMemoryBankDequeue:
    """Test get_queue_embeddings returns correct data."""

    def test_empty_queue_returns_empty(self):
        bank = MemoryBank(queue_size=8, embedding_dim=4)
        imu_q, text_q = bank.get_queue_embeddings()
        assert imu_q.shape == (0, 4)
        assert text_q.shape == (0, 4)

    def test_partial_fill_returns_active_portion(self):
        bank = MemoryBank(queue_size=16, embedding_dim=4)
        imu = torch.randn(5, 4)
        text = torch.randn(5, 4)
        bank.update(imu, text)

        imu_q, text_q = bank.get_queue_embeddings()
        assert imu_q.shape == (5, 4)
        assert text_q.shape == (5, 4)
        assert torch.allclose(imu_q, imu)
        assert torch.allclose(text_q, text)

    def test_dequeue_device_transfer(self):
        bank = MemoryBank(queue_size=8, embedding_dim=4, device=torch.device('cpu'))
        bank.update(torch.randn(4, 4), torch.randn(4, 4))
        imu_q, text_q = bank.get_queue_embeddings(device=torch.device('cpu'))
        assert imu_q.device == torch.device('cpu')

    def test_dequeue_after_wraparound(self):
        """After wraparound, should return full queue_size embeddings."""
        queue_size = 4
        bank = MemoryBank(queue_size=queue_size, embedding_dim=4)
        # Fill and wrap
        bank.update(torch.randn(3, 4), torch.randn(3, 4))
        bank.update(torch.randn(3, 4), torch.randn(3, 4))
        assert bank.is_full
        imu_q, text_q = bank.get_queue_embeddings()
        assert imu_q.shape == (queue_size, 4)


class TestMemoryBankValidation:
    """Test NaN and zero-norm validation."""

    def test_rejects_nan_embeddings(self):
        bank = MemoryBank(queue_size=8, embedding_dim=4)
        imu = torch.randn(4, 4)
        imu[1, 2] = float('nan')
        text = torch.randn(4, 4)
        with pytest.raises(ValueError, match="NaN detected"):
            bank.update(imu, text)

    def test_rejects_zero_norm_embeddings(self):
        bank = MemoryBank(queue_size=8, embedding_dim=4)
        imu = torch.randn(4, 4)
        imu[2] = 0.0  # Zero-norm vector
        text = torch.randn(4, 4)
        with pytest.raises(ValueError, match="Zero-norm"):
            bank.update(imu, text)


class TestMemoryBankStateDict:
    """Test state_dict save/load."""

    def test_roundtrip(self):
        bank = MemoryBank(queue_size=8, embedding_dim=4)
        imu = torch.randn(6, 4)
        text = torch.randn(6, 4)
        bank.update(imu, text)

        state = bank.state_dict()
        bank2 = MemoryBank(queue_size=8, embedding_dim=4)
        bank2.load_state_dict(state)

        assert bank2.ptr == bank.ptr
        assert bank2.is_full == bank.is_full

        imu_q1, text_q1 = bank.get_queue_embeddings()
        imu_q2, text_q2 = bank2.get_queue_embeddings()
        assert torch.allclose(imu_q1, imu_q2)
        assert torch.allclose(text_q1, text_q2)
