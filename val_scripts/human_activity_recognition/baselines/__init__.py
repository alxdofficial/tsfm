"""Baseline adapters for eval v2. Importing this package registers every
adapter into `base.REGISTRY`. Add a new baseline by dropping a module here and
importing it below.

Current set: CrossHAR + LiMU-BERT (ConSE tier). Planned: UniMTS (cosine tier,
released weights) and ssl-wearables / UK-Biobank (ConSE tier) — each a ~80-line
adapter module added here. Dropped from v1: MOMENT, LanHAR, LLaSA.
"""

from .base import (  # noqa: F401
    REGISTRY, register, BaselineAdapter, ConSEAdapter, CosineAdapter,
    load_gt, score, global_labels,
)

# Import adapter modules for their @register side effects.
from . import crosshar        # noqa: F401
from . import limubert        # noqa: F401
from . import ssl_wearables   # noqa: F401
from . import deepconvlstm    # noqa: F401  (fewshot tier — run via run_fewshot_v2.py)
from . import unimts          # noqa: F401  (cosine tier)
from . import normwear        # noqa: F401  (l1 tier)

__all__ = ["REGISTRY", "register", "BaselineAdapter", "ConSEAdapter", "CosineAdapter", "load_gt", "score"]
