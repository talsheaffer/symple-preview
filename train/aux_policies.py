import torch
from src.model.environment import Symple

def random_policy(env: Symple) -> torch.Tensor:
    """
    A random policy that returns equal probabilities for all valid actions.
    
    Args:
    env (Symple): The Symple environment.
    
    Returns:
    torch.Tensor: A tensor of probabilities for each action.
    """
    valid_actions = env.validity_mask.nonzero().squeeze()
    num_valid_actions = valid_actions.size(0)
    
    # Create a tensor of equal probabilities for valid actions
    probs = torch.zeros_like(env.validity_mask, dtype=torch.float32)
    probs[valid_actions] = 1.0 / num_valid_actions
    
    return probs[None,:]

