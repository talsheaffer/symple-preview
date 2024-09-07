
import torch
from src.model.model import SympleAgent
from src.model.environment import Symple
from typing import List

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




def train_on_batch(agent: SympleAgent, envs: List[Symple], optimizer: torch.optim.Optimizer):
    """
    Train the agent on a batch of Symple environment instances.

    Args:
    agent (SympleAgent): The agent to train.
    envs (List[Symple]): A list of Symple environment instances to train on.
    optimizer (torch.optim.Optimizer): The optimizer to use for training.

    Returns:
    float: The average loss for this batch.
    """
    agent.train()
    optimizer.zero_grad()

    total_loss = 0.0
    for env in envs:
        loss = accumulate_gradients(agent, env)
        total_loss += loss

    # Compute average loss
    avg_loss = total_loss / len(envs)

    # Perform optimization step
    optimizer.step()

    return avg_loss
