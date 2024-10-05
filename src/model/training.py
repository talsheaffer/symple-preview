
import torch
from src.model.model import SympleAgent
from src.model.environment import Symple
from src.model.state import SympleState
from typing import List, Callable, Union, Optional, Tuple, Dict, Any

def compute_loss(
        returns: torch.Tensor,
        probs: List[torch.Tensor],
        baseline: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the loss for Monte Carlo training.
    
    Args:
    returns (torch.Tensor): Tensor of returns for each step.
    probs (List[torch.Tensor]): List of probabilities of chosen actions.
    baseline (Optional[torch.Tensor]): Optional baseline to subtract from returns.
    
    Returns:
    torch.Tensor: The computed loss.
    """

    if baseline is None:
        baseline = torch.zeros_like(returns)
    else:
        if len(baseline) < len(returns):
            baseline = torch.nn.functional.pad(baseline, (0, len(returns) - len(baseline)))
        elif len(baseline) > len(returns):
            baseline = baseline[:len(returns)]

    # Compute loss
    action_log_probs = torch.stack(probs).log()
    loss = -((returns-baseline) * action_log_probs).sum()
    
    return loss

def compute_loss_off_policy(
        rewards: List[float],
        target_policy_probs: List[torch.Tensor],
        behavior_policy_probs: List[torch.Tensor],
        gamma: float = 1.,
        baseline: Optional[Union[List[float], float]] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Compute the loss for off-policy Monte Carlo training.
    
    Args:
    rewards (List[float]): List of rewards for each step.
    target_policy_probs (List[torch.Tensor]): List of probabilities of chosen actions under the target policy.
    behavior_policy_probs (List[torch.Tensor]): List of probabilities of chosen actions under the behavior policy.
    gamma (float): Discount factor for future rewards.
    baseline (Optional[Union[List[float], float]]): Optional baseline to subtract from returns.
    
    Returns:
    Tuple[torch.Tensor, float]: The computed loss and the total return.
    """
    # Compute importance sampling ratios
    target_policy_probs = torch.stack(target_policy_probs)
    behavior_policy_probs = torch.stack(behavior_policy_probs)
    importance_ratios = target_policy_probs / behavior_policy_probs.detach() # Detach to prevent gradient flow

    T = len(rewards)

    returns = torch.zeros(T, device=target_policy_probs[0].device)

    if not baseline:
        baseline = returns.clone()
    elif isinstance(baseline, list):
        # pad with zeros if necessary
        if len(baseline) < T:
            baseline = baseline + [0] * (T - len(baseline))
        baseline = torch.tensor(baseline[:T])
    elif isinstance(baseline, float):
        baseline = baseline * torch.ones(T, device = target_policy_probs[0].device)
    
    # Compute discounted returns using importance sampling
    total_return = torch.zeros(1, device=target_policy_probs[0].device)
    for t in reversed(range(T)):
        returns[t] =  importance_ratios[t]*(rewards[t] + gamma * total_return)
        total_return += returns[t].detach()

    loss = -returns.sum()
    
    return loss, total_return.item()

def compute_returns(
        rewards: List[float],
        probs: List[torch.Tensor],
        behavior_policy_probs: Optional[List[torch.Tensor]] = None,
        gamma: float = 1.,
) -> torch.Tensor:
    """
    Compute the returns for each step.

    Args:
    rewards (List[float]): List of rewards for each step.
    probs (List[torch.Tensor]): List of probabilities of chosen actions.
    behavior_policy_probs (Optional[List[torch.Tensor]]): List of probabilities of chosen actions under the behavior policy.
    gamma (float): Discount factor for future rewards.

    Returns:
    torch.Tensor: The computed returns for each step.
    """
    T = len(rewards)
    returns = torch.zeros(T, device=probs[0].device)
    future_return = 0
    for t in reversed(range(T)):
        future_return = (rewards[t] + gamma * future_return)
        if behavior_policy_probs:
            future_return *= (probs[t].item() / behavior_policy_probs[t].item())
        returns[t] = future_return
    return returns


def train_on_batch(
        agent: SympleAgent, 
        states: List[SympleState], 
        optimizer: torch.optim.Optimizer,
        behavior_policy: Optional[Callable[[SympleState, Symple], Union[torch.Tensor, List[float]]]] = None,
        baseline: Optional[Union[List[float], float]] = None,
        gamma: float = 1.,
        agent_forward_kwargs: Dict[str, Any] = {},
        **symple_kwargs: Dict[str, Any]
) -> Tuple[float, List[Dict[str, Any]], List[SympleState], torch.Tensor]:
    """
    Train the agent on a batch of SympleStates.

    Args:
    agent (SympleAgent): The agent to train.
    states (List[SympleState]): A list of SympleState instances to train on.
    optimizer (torch.optim.Optimizer): The optimizer to use for training.
    behavior_policy (Optional[Callable[[SympleState, Symple], Union[torch.Tensor, List[float]]]]): The behavior policy to use for off-policy training if given.
    baseline (Optional[Union[List[float], float]]): Optional baseline to subtract from returns.
    gamma (float): Discount factor for future rewards.
    **symple_kwargs: Keyword arguments to pass to Symple constructor for each SympleState.

    Returns:
    Tuple[float, List[Dict[str, Any]], List[SympleState], torch.Tensor]: 
        - The average return for this batch.
        - The batch history.
        - The output SympleStates from agent.forward.
        - The average returns by step.
    """
    agent.train()
    optimizer.zero_grad()
    batch_history = []
    output_states = []
    avg_returns_by_step = torch.tensor([], device=agent.device)
    num_examples_by_step = torch.tensor([], device=agent.device)

    for state in states:
        env = Symple(**symple_kwargs)
        if behavior_policy:
            history, output_state = agent(state, env, behavior_policy=behavior_policy, **agent_forward_kwargs)
            output_states.append(output_state)
            
            rewards = [step['reward'] for step in history]
            target_action_probs = [step['target_probability'] for step in history]
            behavior_action_probs = [step['behavior_probability'] for step in history]
            returns = compute_returns(rewards, target_action_probs, behavior_action_probs, gamma=gamma)
            probs = target_action_probs
        else:
            history, output_state = agent(state, env, **agent_forward_kwargs)
            output_states.append(output_state)
            rewards = [step['reward'] for step in history]
            returns = compute_returns(rewards, [step['probability'] for step in history], gamma=gamma)
            probs = [step['probability'] for step in history]

        loss = compute_loss(returns, probs, baseline=baseline)
        loss = loss / len(states)
        loss.backward()

        # Update average returns by step
        if len(returns) > len(avg_returns_by_step):
            avg_returns_by_step = torch.nn.functional.pad(avg_returns_by_step, (0, len(returns) - len(avg_returns_by_step)))
            num_examples_by_step = torch.nn.functional.pad(num_examples_by_step, (0, len(returns) - len(num_examples_by_step)))

        for t, ret in enumerate(returns):
            avg_returns_by_step[t] += ret.item()
            num_examples_by_step[t] += 1

        # Convert tensors to items in history
        for step in history:
            for key, value in step.items():
                if isinstance(value, torch.Tensor):
                    step[key] = value.item()
        
        batch_history.append({
            'example_history': history,
            'return': returns[0].item(),
            'loss': loss.item(),
            'node_count_reduction': sum([step['node_count_reduction'] for step in history])
        })

    # Compute average returns by step
    avg_returns_by_step = torch.where(num_examples_by_step > 0, 
                                      avg_returns_by_step / num_examples_by_step, 
                                      torch.zeros_like(avg_returns_by_step))

    # Perform optimization step
    optimizer.step()

    avg_return = avg_returns_by_step[0].item()  # Average return of the first step
    return avg_return, batch_history, output_states, avg_returns_by_step


def update_value_function_estimate(
        current_estimate: torch.Tensor,
        avg_batch_returns: torch.Tensor,
        batch_num: int
) -> torch.Tensor:
    """
    Update the value function estimate using the average batch returns.
    Pad to the longest tensor between current_estimate and avg_batch_returns.

    Args:
    current_estimate (torch.Tensor): The current value function estimate.
    avg_batch_returns (torch.Tensor): The average returns from the current batch.
    batch_num (int): The current batch number.

    Returns:
    torch.Tensor: The updated value function estimate.
    """
    max_length = max(current_estimate.shape[0], avg_batch_returns.shape[0])
    padded_current = torch.nn.functional.pad(current_estimate, (0, max_length - current_estimate.shape[0]))
    padded_avg_returns = torch.nn.functional.pad(avg_batch_returns, (0, max_length - avg_batch_returns.shape[0]))
    
    return padded_current * (batch_num - 1) / batch_num + padded_avg_returns / batch_num


def train_on_batch_with_value_function_baseline(
        *args: Any,
        V: torch.Tensor,
        batch_num: int,
        **kwargs: Any
) -> Tuple[float, List[Dict[str, Any]], List[SympleState], torch.Tensor]:
    """
    Train on a batch of SympleStates with a value function baseline.

    Args:
    *args: Positional arguments to pass to train_on_batch.
    V (torch.Tensor): The current value function estimate.
    batch_num (int): The current batch number.
    **kwargs: Keyword arguments to pass to train_on_batch.

    Returns:
    Tuple[float, List[Dict[str, Any]], List[SympleState], torch.Tensor]:
        - The average return for this batch.
        - The batch history.
        - The output SympleStates from agent.forward.
        - The updated value function estimate.
    """
    avg_return, batch_history, output_states, avg_returns_by_step = train_on_batch(*args, baseline=V, **kwargs)

    V = update_value_function_estimate(V, avg_returns_by_step, batch_num)
    return avg_return, batch_history, output_states, V
