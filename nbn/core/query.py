from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class NBNQuery:
    """A probabilistic query on a BN.

    Parameters
    ----------
    targets:
        Node names whose posterior we want.
    evidence:
        Observed values; shape ``[B, D_node]`` per node.
    do:
        Intervention values (Pearl's do-operator); not scored by likelihood.
    """

    targets: List[str]
    evidence: Dict[str, torch.Tensor] = field(default_factory=dict)
    do: Dict[str, torch.Tensor] = field(default_factory=dict)

    def batch_size(self) -> int:
        """Infer batch dimension B from evidence/do tensors."""
        for v in list(self.evidence.values()) + list(self.do.values()):
            if v.dim() >= 1:
                return v.shape[0]
        return 1
