import torch
from src.model.model import ExprNode

def random_policy(state : ExprNode, validity_mask : torch.Tensor) -> torch.Tensor:
    """
    A random policy that returns equal probabilities for all valid actions.
    
    Args:
    state (ExprNode): The current state of the environment.
    validity_mask (torch.Tensor): A tensor indicating the validity of actions.
    
    Returns:
    torch.Tensor: A tensor of probabilities for each action.
    """
    return validity_mask.view((1,len(validity_mask)))/validity_mask.sum()

