"""M3 — the first controlled DeBERTa-v3-base cache-equivalence fine-tune.

The torch-free layers (``config``, ``cost``, ``selection``, ``report``) import cleanly without the
``train`` optional-extra and carry the offline test suite + the pre-flight gate. The layers that
touch weights (``data``, ``model``, ``runner``, ``predict``) lazy-import torch/transformers inside
functions, exactly like the M2 baseline scorers, so importing the package never needs a GPU.
"""

from __future__ import annotations

__all__: list[str] = []
