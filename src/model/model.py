import torch
import torch.nn.functional as F
from torch import nn

from src.model.environment import NUM_OPS, Symple
from src.model.ffn import FFN
from src.model.tree import INT_NE_TYPE, INT_PO_TYPE, VOCAB_SIZE, ExprNode

from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE

from typing import Callable, Union, List, Tuple, Optional

from numpy import inf
class SympleEmbedding(nn.Module):
    def __init__(
        self,
        *embedargs,
        int_po_type: int = INT_PO_TYPE,
        int_ne_type: int = INT_NE_TYPE,
        **embedkwargs,
    ):
        super(SympleEmbedding, self).__init__()
        self.int_po_type = int_po_type
        self.int_ne_type = int_ne_type
        self.embedding = nn.Embedding(*embedargs, **embedkwargs)

    def forward(self, input: ExprNode) -> ExprNode:
        # t = input.to_tensor()
        input.embedding = self.embedding(torch.tensor(input.type))
        if input.type in (self.int_ne_type, self.int_po_type):
            input.embedding[-1] = input.arg
        input.embedding = input.embedding[None, :]
        input.a = self(input.a) if input.a is not None else None
        input.b = self(input.b) if input.b is not None else None
        return input


class BinaryTreeLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *lstmargs, **lstmkwargs):
        super(BinaryTreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.left_size = self.hidden_size//2
        self.right_size = self.hidden_size - self.left_size
        self.lstm = nn.LSTM(input_size, hidden_size, *lstmargs, **lstmkwargs)

    def forward(self, input: ExprNode, depth=inf) -> ExprNode:
        if depth < 0:
            return input
        input.a, input.b = (
            self(input.a, depth=depth - 1) if input.a is not None else None,
            self(input.b, depth=depth - 1) if input.b is not None else None,
        )
        input.ensure_parenthood()
        _, (input.hidden, input.cell) = self.lstm(
            input.arg_hot,
                (
                    torch.cat(
                        (
                            input.a.hidden[:,: self.left_size] if input.a is not None else torch.zeros((1, self.left_size)),
                            input.b.hidden[:,: self.right_size] if input.b is not None else torch.zeros((1, self.right_size)),
                        ),
                        dim=-1
                    ),
                    torch.cat(
                        (
                            input.a.cell[:,: self.left_size] if input.a is not None else torch.zeros((1, self.left_size)),
                            input.b.cell[:,: self.right_size] if input.b is not None else torch.zeros((1, self.right_size)),
                        ),
                        dim=-1
                    ),
                ),
            )
        return input


class SympleAgent(nn.Module):
    """
    Currently only actor, no critic.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int = VOCAB_SIZE,
        num_ops=NUM_OPS,
        ffn_n_layers: int = 1,
        lstm_n_layers: int = 1,
    ):
        super(SympleAgent, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_ops = num_ops
        self.num_internal_ops = 2 # modify this when we add more internal ops
        self.lstm = BinaryTreeLSTM(self.vocab_size, self.hidden_size, num_layers=lstm_n_layers)
        self.ffn = FFN(self.hidden_size, self.hidden_size, self.hidden_size, n_layers=ffn_n_layers)

        # Perceptrons for high-level and internal decisions
        self.high_level_actor = nn.Linear(self.hidden_size, 2)
        self.internal_actor = nn.Linear(self.hidden_size, self.num_internal_ops)
        self.actor = nn.Linear(self.hidden_size, self.num_ops)

        self.temperature = 3.0


    def policy(self, current_node: ExprNode, validity_mask: torch.Tensor, recursion_depth: int = 2):
        # Initialize a list to store probabilities, only if training
        probs = [] if self.training else None

        # Decide whether to act or further process the expression
        if recursion_depth <= 0:
            high_level_probs = torch.tensor([[0, 1]], device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            high_level_action = 1
        else:
            high_level_logits = self.high_level_actor(current_node.hidden)
            high_level_probs = F.softmax(high_level_logits/self.temperature, dim=-1)
            high_level_action = torch.multinomial(high_level_probs, 1).item()
        
        # Add probability of high-level action if training
        if self.training:
            probs.append(high_level_probs[:,high_level_action])

        if high_level_action == 0:
            # Internal op
            internal_logits = self.internal_actor(current_node.hidden)
            internal_probs = F.softmax(internal_logits/self.temperature, dim=-1)
            internal_action = torch.multinomial(internal_probs, 1).item()
            
            # Add probability of internal action if training
            if self.training:
                probs.append(internal_probs[:,internal_action])

            if internal_action == 0:
                current_node.hidden = self.ffn(current_node.hidden)
            elif internal_action == 1:
                current_node = self.lstm(current_node)
            else:
                raise ValueError(f"Invalid internal action: {internal_action}")

            action_probs, sub_probs, n_internal_actions = self.policy(current_node, validity_mask, recursion_depth - 1)
            if self.training:
                probs.extend(sub_probs)
            n_internal_actions += 1
            return action_probs, probs, n_internal_actions
        
        n_internal_actions = 0
        if self.training:
            probs.append(high_level_probs[:,high_level_action])
        # Apply validity mask to actor weights
        valid_actor_weights = self.actor.weight * validity_mask.unsqueeze(1)
        valid_actor_bias = self.actor.bias * validity_mask
        logits = validity_mask.log() + F.linear(current_node.hidden, valid_actor_weights, valid_actor_bias)
        action_probs = F.softmax(logits / self.temperature, dim=-1)
        return action_probs, probs, n_internal_actions

    def step(self, state: ExprNode, coord: tuple[int, ...], env: Symple):
        current_node = state.get_node(coord)
        action_probs, probs, n_internal_actions = self.policy(current_node, env.get_validity_mask(current_node))
        action = torch.multinomial(action_probs, 1).item()
        new_state, new_coord, reward, done = env.step(state, coord, action)
        reward += n_internal_actions * env.time_penalty
        if self.training:
            probs.append(action_probs[:,action])
        return new_state, new_coord, reward, done, probs

    def forward(self, state: ExprNode, env: Symple,
                behavior_policy: Optional[
                    Callable[
                        [ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]
                    ]
                ] = None
                ) -> Union[
                    Tuple[List[float], List[torch.Tensor], ExprNode],
                    Tuple[List[float], List[torch.Tensor], List[torch.Tensor], ExprNode],
                    ExprNode
                ]:
        if behavior_policy:
            return self.off_policy_forward(state, env, behavior_policy)
        
        # Apply to all nodes in the tree
        state = self.lstm(state)
        
        coord = ()
        done = False
        if self.training:
            rewards = []
            action_log_probs = []

        while not done:
            state, coord, reward, done, probs = self.step(state, coord, env)
            if self.training:
                rewards.append(reward)
                action_log_probs.append(torch.stack(probs).log().sum())
        if self.training:
            return rewards, action_log_probs, state
        else:
            return state
    
    def off_policy_step(self, state: ExprNode, coord: tuple[int, ...], env: Symple,
                        behavior_policy: Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]]
                        ) -> Tuple[ExprNode, tuple[int, ...], float, bool, float, float]:
        current_node = state.get_node(coord)
        action_probs, _, _ = self.policy(current_node, env.get_validity_mask(current_node)) # ignore n_internal_actions and accumulated_prob - perhaps change this later
        behavior_probs = behavior_policy(state, coord, env)
        action = torch.multinomial(torch.tensor(behavior_probs), 1).item()
        new_state, new_coord, reward, done = env.step(state, coord, action)
        return new_state, new_coord, reward, done, action_probs[:,action], behavior_probs[:,action]
    
    def off_policy_forward(self, state: ExprNode, env: Symple,
                           behavior_policy: Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]]
                           ) -> Union[Tuple[List[float], List[torch.scalar_tensor], List[torch.scalar_tensor], ExprNode], ExprNode]:
        # Apply to all nodes in the tree
        state = self.lstm(state)
        done = False
        coord = ()
        if self.training:
            rewards = []
            action_probs = []
            behavior_action_probs = []
        while not done:
            state, coord, reward, done, action_prob, behavior_action_prob = self.off_policy_step(state, coord, env, behavior_policy)
            if self.training:
                rewards.append(reward)
                action_probs.append(action_prob)
                behavior_action_probs.append(behavior_action_prob)
        if self.training:
            return rewards, action_probs, behavior_action_probs, state
        else:
            return state