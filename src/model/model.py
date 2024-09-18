import torch
import torch.nn.functional as F
from torch import nn

from src.model.environment import NUM_OPS, Symple
from src.model.ffn import FFN
from src.model.tree import VOCAB_SIZE, ExprNode
from src.model.nary_tree_lstm import NaryTreeLSTM

from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE

from typing import Callable, Union, List, Tuple, Optional, Dict

from numpy import inf



class SympleAgent(nn.Module):
    """
    Currently only actor, no critic.
    """

    def __init__(
        self,
        hidden_size: int,
        global_hidden_size: Optional[int] = None,
        # vocab_size: int = VOCAB_SIZE,
        num_ops=NUM_OPS,
        num_internal_ops: int = 5,
        ffn_n_layers: int = 1,
        lstm_n_layers: int = 1,
    ):
        super(SympleAgent, self).__init__()
        self.hidden_size = hidden_size
        self.global_hidden_size = global_hidden_size if global_hidden_size is not None else hidden_size
        self.vocab_size = VOCAB_SIZE
        self.num_ops = num_ops
        self.num_internal_ops = num_internal_ops

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
            input.a = self.apply_binary_lstm(input.a, depth=depth - 1) if input.a is not None else None
            input.b = self.apply_binary_lstm(input.b, depth=depth - 1) if input.b is not None else None
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
            input.a = self.apply_ternary_lstm(input.a, depth=depth - 1) if input.a is not None else None
            input.b = self.apply_ternary_lstm(input.b, depth=depth - 1) if input.b is not None else None
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
        history = []

        # Concatenate current node's hidden state with global hidden state
        features = current_node.hidden

        # Decide whether to act or further process the expression
        if recursion_depth <= 0:
            high_level_probs = torch.tensor([[0, 1]], device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            high_level_action = 1
            high_level_action_prob = high_level_probs[:,high_level_action]
        else:
            high_level_logits = self.high_level_actor(features)
            high_level_probs = F.softmax(high_level_logits/self.temperature, dim=-1)
            high_level_action = torch.multinomial(high_level_probs, 1).item()
            high_level_action_prob = high_level_probs[:,high_level_action]
        
        # Add high-level action to history
        history.append({
            'action_type': 'high_level',
            'action': high_level_action,
            'probability': high_level_action_prob if self.training else high_level_action_prob.item(),
            'complexity': 0.0,
            'node_count_reduction': 0.0
        })

        if high_level_action == 0:
            # Internal op
            internal_logits = self.internal_actor(features)
            internal_probs = F.softmax(internal_logits/self.temperature, dim=-1)
            internal_action = torch.multinomial(internal_probs, 1).item()
            internal_action_prob = internal_probs[:,internal_action]
            
            internal_op, complexity_func = self.internal_ops[internal_action]
            current_node = internal_op(current_node)
            complexity = complexity_func(current_node)
            
            history.append({
                'action_type': 'internal',
                'action': internal_action,
                'probability': internal_action_prob if self.training else internal_action_prob.item(),
                'complexity': complexity,
                'node_count_reduction': 0.0
            })
            
            action_probs, sub_history = self.policy(current_node, validity_mask, recursion_depth - 1)
            history.extend(sub_history)
            return action_probs, history
        
        # Apply validity mask to actor weights
        valid_actor_weights = self.actor.weight * validity_mask.unsqueeze(1)
        valid_actor_bias = self.actor.bias * validity_mask
        logits = validity_mask.log() + F.linear(features, valid_actor_weights, valid_actor_bias)
        action_probs = F.softmax(logits / self.temperature, dim=-1)
        return action_probs, history

    def step(self, state: ExprNode, coord: tuple[int, ...], env: Symple):
        current_node = state.get_node(coord)
        action_probs, history = self.policy(current_node, env.get_validity_mask(current_node))
        action = torch.multinomial(action_probs, 1).item()
        action_prob = action_probs[:,action]

        new_state, new_coord, reward, node_count_reduction, done = env.step(state, coord, action)
        
        # Add reward and coordinates to internal and high-level action history
        for entry in history:
            entry['reward'] = env.time_penalty  - env.compute_penalty_coefficient * entry['complexity']
            entry['coordinates'] = coord
        
        history.append({
            'action_type': 'external',
            'action': action,
            'probability': action_prob if self.training else action_prob.item(),
            'complexity': 0.0,
            'node_count_reduction': node_count_reduction, 
            'reward': reward,
            'coordinates': coord
        })
        
        return new_state, new_coord, done, history

    def forward(self, state: ExprNode, env: Symple,
                behavior_policy: Optional[
                    Callable[
                        [ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]
                    ]
                ] = None
                ) -> Union[
                    Tuple[List[Dict], ExprNode],
                    ExprNode
                ]:
        if behavior_policy:
            return self.off_policy_forward(state, env, behavior_policy)
        
        # Apply to all nodes in the tree
        state = self.apply_binary_lstm(state)
        
        coord = ()
        done = False
        history = []

        while not done:
            state, coord, done, step_history = self.step(state, coord, env)
            history.extend(step_history)

        return history, state.reset_tensors()
    
    def off_policy_step(self, state: ExprNode, coord: tuple[int, ...], env: Symple,
                        behavior_policy: Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]]
                        ) -> Tuple[ExprNode, tuple[int, ...], bool, Dict]:
        current_node = state.get_node(coord)
        validity_mask = env.get_validity_mask(current_node)
        target_probs, policy_history = self.policy(current_node, validity_mask)
        
        behavior_probs = behavior_policy(state, validity_mask)
        action = torch.multinomial(torch.tensor(behavior_probs), 1).item()
        target_action_prob = target_probs[:,action]
        behavior_action_prob = behavior_probs[:,action]
        new_state, new_coord, reward, node_count_reduction, done = env.step(state, coord, action)
        
        for entry in policy_history:
            entry['reward'] = env.time_penalty - (env.compute_penalty_coefficient * entry['complexity'])
            entry['coordinates'] = coord
        
        step_history = {
            'action': action,
            'target_probability': target_action_prob if self.training else target_action_prob.item(),
            'behavior_probability': behavior_action_prob.detach() if self.training else behavior_action_prob.item(),
            'reward': reward,
            'node_count_reduction': node_count_reduction,
            'target_policy_history': policy_history,
            'coordinates': coord
        }
        
        return new_state, new_coord, done, step_history
    
    def off_policy_forward(self, state: ExprNode, env: Symple,
                           behavior_policy: Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]]
                           ) -> Tuple[List[Dict], ExprNode]:
        
        # Apply to all nodes in the tree
        state = self.apply_binary_lstm(state)
        done = False
        coord = ()
        history = []
        
        while not done:
            state, coord, done, step_history = self.off_policy_step(state, coord, env, behavior_policy)
            history.append(step_history)
        
        return history, state.reset_tensors()