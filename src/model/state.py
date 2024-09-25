from dataclasses import dataclass
from typing import Optional

from src.model.tree import ExprNode

from torch import Tensor

@dataclass
class SympleState:
    en: ExprNode
    coord: tuple[int, ...]
    h_glob: Tensor
    c_glob: Tensor
    nc: Optional[int] = None