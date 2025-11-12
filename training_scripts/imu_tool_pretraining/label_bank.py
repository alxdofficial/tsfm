"""
Label bank for caching text embeddings.

Efficiently manages text-to-embedding mappings using SentenceBERT,
with caching to avoid re-encoding the same labels.
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional


class LabelBank:
    """Caches text embeddings to avoid redundant encoding."""

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_name: SentenceBERT model name
            device: Device to run the model on
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache = {}  # {label_text: embedding}

    def encode(self, label_texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode a batch of label texts, using cache when possible.

        Args:
            label_texts: List of text labels to encode
            normalize: Whether to L2-normalize embeddings

        Returns:
            Tensor of shape (len(label_texts), embedding_dim)
        """
        # Check which texts need encoding
        to_encode = []
        to_encode_indices = []

        for i, text in enumerate(label_texts):
            if text not in self.cache:
                to_encode.append(text)
                to_encode_indices.append(i)

        # Encode new texts
        if to_encode:
            embeddings = self.model.encode(
                to_encode,
                convert_to_tensor=True,
                normalize_embeddings=normalize,
                device=self.device
            )

            # Cache new embeddings
            for text, emb in zip(to_encode, embeddings):
                self.cache[text] = emb.cpu()  # Store on CPU to save GPU memory

        # Retrieve all embeddings from cache
        result = torch.stack([self.cache[text] for text in label_texts])
        return result.to(self.device)

    def encode_unique(
        self,
        label_texts: List[str],
        normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode only unique labels and return mapping to original positions.

        Useful for batch processing where multiple samples share labels.

        Args:
            label_texts: List of text labels (may contain duplicates)
            normalize: Whether to L2-normalize embeddings

        Returns:
            embeddings: Tensor of unique embeddings (num_unique, embedding_dim)
            indices: Tensor mapping original positions to unique embeddings

        Example:
            label_texts = ["walk", "run", "walk"]
            embeddings, indices = encode_unique(label_texts)
            # embeddings.shape = (2, 384)  # Only "walk" and "run"
            # indices = [0, 1, 0]  # Maps back to original order
            # reconstructed = embeddings[indices]  # Shape (3, 384)
        """
        # Get unique labels and their indices
        unique_labels = []
        label_to_idx = {}
        indices = []

        for text in label_texts:
            if text not in label_to_idx:
                label_to_idx[text] = len(unique_labels)
                unique_labels.append(text)
            indices.append(label_to_idx[text])

        # Encode unique labels
        embeddings = self.encode(unique_labels, normalize=normalize)
        indices_tensor = torch.tensor(indices, device=self.device)

        return embeddings, indices_tensor

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()

    def cache_size(self) -> int:
        """Return number of cached embeddings."""
        return len(self.cache)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
