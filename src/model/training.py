
import torch
from src.model.model import SympleAgent
from src.model.environment import Symple
from src.model.tree import ExprNode
from typing import List, Callable, Union, Optional, Tuple, Dict, Any

def compute_loss(rewards: List[float], action_log_probs: List[torch.Tensor], gamma: float = 1.) -> Tuple[torch.Tensor, float]:
    """
    Compute the loss for Monte Carlo training.
    
    Args:
    rewards (List[float]): List of rewards for each step.
    action_log_probs (List[torch.Tensor]): List of log probabilities of chosen actions.
    gamma (float): Discount factor for future rewards.
    
    Returns:
    torch.Tensor: The computed loss.
    float: The total return.
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
    
    return loss, returns[0].item()

def compute_loss_off_policy(
        rewards: List[float], target_policy_probs: List[torch.Tensor], behavior_policy_probs: List[torch.Tensor], gamma: float = 1.
) -> Tuple[torch.Tensor, float]:
    """
    Compute the loss for off-policy Monte Carlo training.
    
    Args:
    rewards (List[float]): List of rewards for each step.
    target_policy_probs (List[torch.Tensor]): List of probabilities of chosen actions under the target policy.
    behavior_policy_probs (List[torch.Tensor]): List of probabilities of chosen actions under the behavior policy.
    gamma (float): Discount factor for future rewards.
    
    Returns:
    torch.Tensor: The computed loss.
    float: The total return.
    """
    # Compute importance sampling ratios
    target_policy_probs = torch.stack(target_policy_probs)
    behavior_policy_probs = torch.stack(behavior_policy_probs)
    importance_ratios = target_policy_probs / behavior_policy_probs.detach() # Detach to prevent gradient flow
    
    # Compute discounted returns using importance sampling
    total_return = torch.zeros(1, device=target_policy_probs[0].device)
    for t in reversed(range(len(rewards))):
        total_return = importance_ratios[t] * (rewards[t] + gamma * total_return)

    loss = -total_return
    
    return loss, total_return.item()

def train_on_batch(
        agent: SympleAgent, expr_nodes: List[ExprNode], optimizer: torch.optim.Optimizer,
        behavior_policy: Optional[Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]]] = None,
        **symple_kwargs: Dict[str, Any]
) -> float:
    """
    Train the agent on a batch of ExprNodes.

    Args:
    agent (SympleAgent): The agent to train.
    expr_nodes (List[ExprNode]): A list of ExprNode instances to train on.
    optimizer (torch.optim.Optimizer): The optimizer to use for training.
    behavior_policy (Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]]): The behavior policy to use for off-policy training if given.
    **symple_kwargs: Keyword arguments to pass to Symple constructor for each ExprNode.

    Returns:
    float: The average return for this batch.
    """
    agent.train()
    optimizer.zero_grad()

    avg_return = 0.0
    for en in expr_nodes:
        env = Symple(**symple_kwargs)
        if behavior_policy:
            rewards, action_probs, behavior_action_probs, _ = agent(en, env, behavior_policy=behavior_policy)
            loss, total_return = compute_loss_off_policy(rewards, action_probs, behavior_action_probs)
            loss = loss / len(expr_nodes)
        else:
            rewards, action_log_probs, _ = agent(en, env)
            loss, total_return = compute_loss(rewards, action_log_probs)
            loss = loss / len(expr_nodes)
        
        loss.backward()
        avg_return += total_return / len(expr_nodes)

    # Perform optimization step
    optimizer.step()

    return avg_return
