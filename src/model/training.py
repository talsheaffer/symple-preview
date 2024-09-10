
import torch
from src.model.model import SympleAgent
from src.model.environment import Symple
from typing import List, Callable, Union, Optional

def compute_loss(rewards: List[float], action_log_probs: List[torch.Tensor], gamma: float = 1.) -> torch.Tensor:
    """
    Compute the loss for Monte Carlo training.
    
    Args:
    rewards (List[float]): List of rewards for each step.
    action_log_probs (List[torch.Tensor]): List of log probabilities of chosen actions.
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

    
    # Compute loss
    action_log_probs = torch.stack(action_log_probs)
    loss = -(returns * action_log_probs).sum()
    
    return loss

def compute_loss_off_policy(rewards: List[float], target_policy_probs: List[torch.Tensor], behavior_policy_probs: List[torch.Tensor], gamma: float = 1.) -> torch.Tensor:
    """
    Compute the loss for off-policy Monte Carlo training.
    
    Args:
    rewards (List[float]): List of rewards for each step.
    target_policy_probs (List[torch.Tensor]): List of probabilities of chosen actions under the target policy.
    behavior_policy_probs (List[torch.Tensor]): List of probabilities of chosen actions under the behavior policy.
    gamma (float): Discount factor for future rewards.
    
    Returns:
    torch.Tensor: The computed loss.
    """
    T = len(rewards)
    returns = torch.zeros(T, device=target_policy_probs[0].device)
    
    # Compute discounted returns
    future_return = 0
    for t in reversed(range(T)):
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return

    # Compute importance sampling ratios
    target_policy_probs = torch.stack(target_policy_probs)
    behavior_policy_probs = torch.stack(behavior_policy_probs)
    importance_ratios = target_policy_probs / behavior_policy_probs

    # Compute loss using importance sampling
    loss = -(returns * importance_ratios ).sum()
    
    return loss


def accumulate_gradients(agent: SympleAgent, env: Symple) -> float:
    """
    Perform a forward pass and accumulate gradients without taking an optimizer step.
    
    Args:
    agent (SympleAgent): The agent to train.
    env (Symple): The environment to train on.
    
    Returns:
    float: The loss for this training step.
    """
    agent.train()
    
    rewards, action_log_probs, _ = agent(env)
    loss = compute_loss(rewards, action_log_probs)
    
    loss.backward()
    
    return loss.item()




def train_on_batch(
        agent: SympleAgent, envs: List[Symple], optimizer: torch.optim.Optimizer,
        behavior_policy: Optional[Callable[[Symple], Union[torch.Tensor, List[float]]]] = None,
) -> float:
    """
    Train the agent on a batch of Symple environment instances.

    Args:
    agent (SympleAgent): The agent to train.
    envs (List[Symple]): A list of Symple environment instances to train on.
    optimizer (torch.optim.Optimizer): The optimizer to use for training.
    behavior_policy (Callable[[Symple], Union[torch.Tensor, List[float]]]): The behavior policy to use for off-policy training if given.

    Returns:
    float: The average loss for this batch.
    """
    agent.train()
    optimizer.zero_grad()

    avg_loss = 0.0
    for env in envs:
        if behavior_policy:
            rewards, action_probs, behavior_action_probs, _ = agent(env, behavior_policy = behavior_policy)
            loss = compute_loss_off_policy(rewards, action_probs, behavior_action_probs) / len(envs)
        else:
            rewards, action_log_probs, _ = agent(env)
            loss = compute_loss(rewards, action_log_probs) / len(envs)
        loss.backward()
        loss = loss.item()
        avg_loss += loss

    # Perform optimization step
    optimizer.step()

    return avg_loss
