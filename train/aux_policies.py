import torch
from src.model.model import SympleState
from typing import Tuple

def random_policy(state: SympleState, validity_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A random policy that returns equal probabilities for all valid actions.
    
    Args:
    state (SympleState): The current state of the environment.
    validity_mask (torch.Tensor): A tensor indicating the validity of external actions.
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tensors of probabilities for high-level, external, and teleport actions.
    """
    # High-level actions: external, teleport, finish
    p_high = torch.ones((1,3), device=validity_mask.device) / 3
    
    p_ext = validity_mask[None, :].float() / validity_mask.sum()
    
    # Assuming teleport action is always valid for all nodes
    p_teleport = torch.ones((1, state.en.node_count()), device=validity_mask.device) / state.en.node_count()
    
    return p_high, p_ext, p_teleport
