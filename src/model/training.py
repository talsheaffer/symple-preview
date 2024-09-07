
import torch
from src.model.model import SympleAgent
from src.model.environment import Symple

def compute_loss(rewards, action_log_probs, gamma=1.):
    """
    Compute the loss for Monte Carlo training.
    
    Args:
    rewards (list): List of rewards for each step.
    action_log_probs (list): List of log probabilities of chosen actions.
    gamma (float): Discount factor for future rewards.
    
    Returns:
    torch.Tensor: The computed loss.
    """
    T = len(rewards)
    returns = torch.zeros(T, device=action_log_probs[0].device)
    
    # Compute discounted returns
    future_return = 0
    for t in reversed(range(T)):
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return
    
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Compute loss
    action_log_probs = torch.stack(action_log_probs)
    loss = -(returns * action_log_probs).sum()
    
    return loss

def accumulate_gradients(agent: SympleAgent, env: Symple):
    """
    Perform a forward pass and accumulate gradients without taking an optimizer step.
    
    Args:
    agent (SympleAgent): The agent to train.
    env (Symple): The environment to train on.
    
    Returns:
    float: The loss for this training step.
    """
    agent.train()
    
    rewards, action_log_probs = agent(env)
    loss = compute_loss(rewards, action_log_probs)
    
    loss.backward()
    
    return loss.item()




