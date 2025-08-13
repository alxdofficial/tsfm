import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datasets.BaseDataset import BaseEpisodesDataset

# ==================================
# ActivityClsDataset (with labels)
# ==================================
class ActionSenseActivityClsDataset(BaseEpisodesDataset):
    """
    Activity classification pretraining dataset.
    Assumes the converter attached labels either as:
      - a constant DataFrame column "__activity__" per row (recommended), or
      - df.attrs["activity"] (constant per segment).

    Policy:
      - Use the label of the CENTRAL patch in the context window.
      - If the episode has no labels, raises a clear error at init (unless allow_unlabeled=True).
    """
    def __init__(
        self,
        episodes: List[pd.DataFrame],
        metadata: Dict,
        context_size: int = -1,
        debug: bool = True,
        store_episode_stats: bool = False,
        allow_unlabeled: bool = False,
    ):
        self.allow_unlabeled = allow_unlabeled
        super().__init__(
            episodes=episodes,
            metadata=metadata,
            context_size=context_size,
            debug=debug,
            store_episode_stats=store_episode_stats,
        )
        # Build activity_to_id from all patches that have labels
        all_labels: List[str] = []
        for epi in self.episodes:
            if "__activity__" in epi:
                all_labels.extend(epi["__activity__"])
        unique = sorted(set(all_labels))
        if not unique and not allow_unlabeled:
            raise ValueError(
                "No activity labels found in episodes. "
                "Ensure your converter sets df['__activity__'] or df.attrs['activity'], "
                "or initialize with allow_unlabeled=True for debugging."
            )
        self.activity_to_id: Dict[str, int] = {name: i for i, name in enumerate(unique)}
        self.num_classes: int = len(self.activity_to_id)
        if self.debug:
            print(f"[DEBUG] ActivityClsDataset: discovered {self.num_classes} classes.")

        self.id_to_activity = [name for name,_ in sorted(self.activity_to_id.items(), key=lambda kv: kv[1])]
        print("[DBG] classes:", self.id_to_activity)


    def get_target(
        self,
        epi_idx: int,
        central_patch_idx: int,
        ctx_start: int,
        ctx_end_inclusive: int,
        context_payload: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        labels = context_payload.get("__activity__", None)
        if labels is None:
            if self.allow_unlabeled:
                return None
            raise KeyError(
                f"Episode {epi_idx} has no '__activity__'. "
                f"Set allow_unlabeled=True or ensure the converter attaches labels."
            )
        # Central patch determines the label for this sample
        central_label = labels[central_patch_idx]
        cls_id = self.activity_to_id.get(central_label, None)
        if cls_id is None:
            # Unknown label (unlikely if mapping built from episodes)
            if self.allow_unlabeled:
                return None
            raise KeyError(f"Unknown activity label '{central_label}' at episode {epi_idx}, patch {central_patch_idx}.")
        return {"activity_id": torch.tensor(cls_id, dtype=torch.long)}

# =========================
# MSPDataset (no targets)
# =========================
class ActionSenseMSPDataset(BaseEpisodesDataset):
    """
    Masked Self-Prediction pretraining dataset.
    - Inherits all base logic for slicing/stacking patches.
    - Returns no targets (encoder builds them internally).
    """
    def get_target(
        self,
        epi_idx: int,
        central_patch_idx: int,
        ctx_start: int,
        ctx_end_inclusive: int,
        context_payload: Dict[str, Any],
    ) -> Optional[Dict[str, torch.Tensor]]:
        # MSP needs no labels; keep "target": None as in your original.
        return None
