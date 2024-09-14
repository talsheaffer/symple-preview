import torch
import torch.nn.functional as F
from torch import nn

from src.model.environment import NUM_OPS, Symple
from src.model.ffn import FFN
from src.model.tree import VOCAB_SIZE, ExprNode
from src.model.nary_tree_lstm import NaryTreeLSTM

from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE

from typing import Callable, Union, List, Tuple, Optional

from numpy import inf



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

        # nn modules
        # self.lstm = BinaryTreeLSTM(self.vocab_size, self.hidden_size, num_layers=lstm_n_layers)
        self.blstm = NaryTreeLSTM(2, self.vocab_size, self.hidden_size)
        self.tlstm = NaryTreeLSTM(3, self.vocab_size, self.hidden_size)
        self.ffn = FFN(self.hidden_size, self.hidden_size, self.hidden_size, n_layers=ffn_n_layers)

        # Perceptrons for high-level and internal decisions
        self.high_level_actor = nn.Linear(self.hidden_size, 2)
        self.internal_actor = nn.Linear(self.hidden_size, self.num_internal_ops)
        self.actor = nn.Linear(self.hidden_size, self.num_ops)

        # internal ops and their compute complexity
        self.internal_ops = {
            0: (self.apply_ffn, lambda en: self.hidden_size**2 * self.ffn.n_layers), 
        }
        for i in range(self.num_internal_ops-1):
            self.internal_ops[i+1] = (
                lambda en: self.apply_ternary_lstm(en, depth = i), lambda en: 4 * self.hidden_size**2 * 3 * en.node_count(depth=i)
            )
        # auxiliary variables
        self.temperature = 3.0
        self.tz = torch.zeros((1, self.hidden_size), device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)

    def apply_ffn(self, input: ExprNode) -> ExprNode:
        input.hidden = self.ffn(input.hidden)
        return input
    
    def apply_binary_lstm(self, input: ExprNode, depth=inf) -> ExprNode:
        """
        Applies a binary LSTM to the input expression tree recursively.

        Args:
            input (ExprNode): The root node of the expression tree.
            depth (int, optional): The maximum depth to traverse. Defaults to infinity.

        Returns:
            ExprNode: The input node with updated hidden and cell states.
        """
        if depth > 0:
            input.a, input.b = (
                self.apply_binary_lstm(input.a, depth=depth - 1) if input.a is not None else None,
                self.apply_binary_lstm(input.b, depth=depth - 1) if input.b is not None else None,
            )
            input.ensure_parenthood()

        (input.hidden, input.cell) = self.blstm(
            input.arg_hot,
            (
                input.a.hidden if input.a is not None else self.tz,
                input.b.hidden if input.b is not None else self.tz,
            ),
            (
                input.a.cell if input.a is not None else self.tz,
                input.b.cell if input.b is not None else self.tz,
            ),
        )
        return input
    

    def apply_ternary_lstm(self, input: ExprNode, depth=0) -> ExprNode:
        """
        Applies a ternary LSTM to the input expression tree recursively.

        Args:
            input (ExprNode): The root node of the expression tree.
            depth (int, optional): The maximum depth to traverse. Defaults to 0.

        Returns:
            ExprNode: The input node with updated hidden and cell states.
        """
        if depth > 0:
            input.a, input.b = (
                self.apply_ternary_lstm(input.a, depth=depth - 1) if input.a is not None else None,
                self.apply_ternary_lstm(input.b, depth=depth - 1) if input.b is not None else None,
            )
        input.ensure_parenthood()

        (input.hidden, input.cell) = self.tlstm(
            input.arg_hot,
            (
                input.hidden if input.a is not None else self.tz,
                input.a.hidden if input.a is not None else self.tz,
                input.b.hidden if input.b is not None else self.tz,
            ),
            (
                input.cell if input.a is not None else self.tz,
                input.a.cell if input.a is not None else self.tz,
                input.b.cell if input.b is not None else self.tz,
            )
        )
        return input
        

    def policy(self, current_node: ExprNode, validity_mask: torch.Tensor, recursion_depth: int = 5):
        # Initialize a list to store probabilities, only if training
        probs = [] if self.training else None
        complexities = [] if self.training else None

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
            complexities.append(0.0)

        if high_level_action == 0:
            # Internal op
            internal_logits = self.internal_actor(current_node.hidden)
            internal_probs = F.softmax(internal_logits/self.temperature, dim=-1)
            internal_action = torch.multinomial(internal_probs, 1).item()
            
            internal_op, complexity_func = self.internal_ops[internal_action]
            current_node = internal_op(current_node)
            action_probs, sub_probs, sub_complexities = self.policy(current_node, validity_mask, recursion_depth - 1)
            if self.training:
                probs.append(internal_probs[:,internal_action])
                complexities.append(complexity_func(current_node))
                probs.extend(sub_probs)
                complexities.extend(sub_complexities)
            return action_probs, probs, complexities
        
        # Apply validity mask to actor weights
        valid_actor_weights = self.actor.weight * validity_mask.unsqueeze(1)
        valid_actor_bias = self.actor.bias * validity_mask
        logits = validity_mask.log() + F.linear(current_node.hidden, valid_actor_weights, valid_actor_bias)
        action_probs = F.softmax(logits / self.temperature, dim=-1)
        return action_probs, probs, complexities

    def step(self, state: ExprNode, coord: tuple[int, ...], env: Symple):
        current_node = state.get_node(coord)
        action_probs, probs, complexities = self.policy(current_node, env.get_validity_mask(current_node))
        if self.training:
            rewards = [env.time_penalty - env.compute_penalty_coefficient * c for c in complexities]
        action = torch.multinomial(action_probs, 1).item()
        new_state, new_coord, reward, done = env.step(state, coord, action)
        if self.training:
            probs.append(action_probs[:,action])
            rewards.append(reward)
        return new_state, new_coord, rewards, done, probs

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
        state = self.apply_binary_lstm(state)
        coord = ()
        done = False
        if self.training:
            rewards = []
            action_log_probs = []

        while not done:
            state, coord, new_rewards, done, probs = self.step(state, coord, env)
            if self.training:
                rewards.extend(new_rewards)
                action_log_probs.extend(p.log() for p in probs)
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