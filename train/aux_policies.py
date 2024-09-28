import torch
from src.model.model import SympleState

def random_policy(state: SympleState, validity_mask: torch.Tensor) -> torch.Tensor:
    """
    A random policy that returns equal probabilities for all valid actions.
    
    Args:
    state (SympleState): The current state of the environment.
    validity_mask (torch.Tensor): A tensor indicating the validity of actions.
    
    Returns:
    torch.Tensor: A tensor of probabilities for each action.
    """
    return validity_mask.view((1, len(validity_mask))) / validity_mask.sum()
