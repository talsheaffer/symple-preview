import torch
import torch.nn.functional as F
from torch import nn

from src.model.environment import NUM_OPS, Symple, SympleState
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
        self.lstm = nn.LSTM(self.hidden_size, self.global_hidden_size, num_layers=lstm_n_layers, batch_first=True)
        self.blstm = NaryTreeLSTM(2, self.vocab_size, self.hidden_size)
        self.tlstm = NaryTreeLSTM(3, self.vocab_size, self.hidden_size)
        self.ffn = FFN(self.hidden_size, self.hidden_size, self.hidden_size, n_layers=ffn_n_layers)
        self.teleport_ffn = FFN(self.global_hidden_size + self.hidden_size, self.hidden_size, 1, n_layers=ffn_n_layers)

        # FFNs for high-level and internal decisions
        self.high_level_actor = FFN(self.hidden_size + self.global_hidden_size, self.hidden_size, 4, n_layers=ffn_n_layers)  # Changed to 4 for teleport and finish actions
        self.internal_actor = FFN(self.hidden_size + self.global_hidden_size, self.hidden_size, self.num_internal_ops, n_layers=ffn_n_layers)
        self.actor = FFN(self.hidden_size + self.global_hidden_size, self.hidden_size, self.num_ops, n_layers=ffn_n_layers)

        # internal ops and their compute complexity. Including certain compositions of elementary internal ops
        self.ffn_complexity = self.hidden_size**2 * self.ffn.n_layers
        self.glstm_complexity = 4 * (self.global_hidden_size * (self.hidden_size + self.global_hidden_size) * self.lstm.num_layers)
        self.blstm_complexity = 4 * (self.hidden_size * (self.vocab_size + self.hidden_size) * 2) 
        self.tlstm_complexity = 4 * (self.hidden_size * (self.vocab_size + self.hidden_size) * 3)
       
        self.internal_ops = [
            (lambda en, h, c: (self.apply_ffn(en), h, c), lambda en: self.ffn_complexity),
            (
                lambda en, h, c: self.apply_global_lstm(self.apply_ffn(en), h, c),
                lambda en: self.ffn_complexity + self.glstm_complexity
            ),
            (self.apply_global_lstm, lambda en: self.glstm_complexity)
        ]
        for i in range(self.num_internal_ops-3):
            if i %2 == 0:
                self.internal_ops.append(
                    (
                        lambda en, h, c: (self.apply_ternary_lstm(en, depth = i//2), h, c),
                        lambda en: self.tlstm_complexity * en.node_count(depth=i//2)
                    )
                )
            else:
                self.internal_ops.append(
                    (
                        lambda en, h, c: self.apply_global_lstm(self.apply_ternary_lstm(en, depth = i//2), h, c),
                        lambda en: self.tlstm_complexity * en.node_count(depth=i//2) + self.glstm_complexity
                    )
                )
        # auxiliary variables
        self.device = DEFAULT_DEVICE
        
    @property
    def tz(self):
        return torch.zeros((1, self.hidden_size), device=self.device, dtype=DEFAULT_DTYPE)

    def apply_internal_op(self, input: ExprNode, op_num: int, h_glob: torch.Tensor, c_glob: torch.Tensor) -> Tuple[ExprNode, float, torch.Tensor, torch.Tensor]:
        """
        Applies an internal operation to the input expression tree recursively.

        Args:
            input (ExprNode): The root node of the expression tree.
            op_num (int): The number of the internal operation to apply.
            h_glob (torch.Tensor): Global hidden state.
            c_glob (torch.Tensor): Global cell state.
        Returns:
            Tuple[ExprNode, float, torch.Tensor, torch.Tensor]: The input node with updated hidden and cell states, and updated global states.
            The float is the complexity of the operation.
        """
        op, complexity_func = self.internal_ops[op_num]
        input, h_glob, c_glob = op(input, h_glob, c_glob)
        complexity = complexity_func(input)
        return input, complexity, h_glob, c_glob
    
    def apply_global_lstm(self, input: ExprNode, h_glob: torch.Tensor, c_glob: torch.Tensor) -> Tuple[ExprNode, torch.Tensor, torch.Tensor]:
        """
        Applies a global LSTM to the input expression tree recursively.

        Args:
            input (ExprNode): The root node of the expression tree.
            h_glob (torch.Tensor): Global hidden state.
            c_glob (torch.Tensor): Global cell state.

        Returns:
            Tuple[ExprNode, torch.Tensor, torch.Tensor]: The input node with updated hidden and cell states, and updated global states.
        """
        _, (h_glob, c_glob) = self.lstm(input.hidden.unsqueeze(0), (h_glob, c_glob))
        return input, h_glob, c_glob
    
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
    
    def apply_high_level_perceptron(self, features: torch.Tensor, recursion_depth: int = 5, temperature: Optional[float] = None, high_level_mask: Optional[torch.Tensor] = None) -> Tuple[dict, int]:
        if recursion_depth <= 0:
            high_level_probs = torch.tensor([[0, 0, 1, 0]], device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)
            high_level_action = 2
            high_level_action_prob = high_level_probs[:,high_level_action]
            target_high_level_action_prob = high_level_action_prob
        else:
            high_level_logits = self.high_level_actor(features)
            if temperature is not None:
                target_probs = F.softmax(high_level_logits, dim=-1)
                high_level_probs = F.softmax(high_level_logits/temperature, dim=-1)
            else:
                high_level_probs = F.softmax(high_level_logits, dim=-1)
            if high_level_mask is None:
                high_level_action = torch.multinomial(high_level_probs, 1).item()
            else:
                masked_high_level_probs = high_level_probs * high_level_mask
                masked_high_level_probs = masked_high_level_probs / masked_high_level_probs.sum(dim=-1, keepdim=True)
                high_level_action = torch.multinomial(masked_high_level_probs, 1).item()
            high_level_action_prob = high_level_probs[:,high_level_action]
            if temperature is not None:
                target_high_level_action_prob = target_probs[:,high_level_action]
        
        # Add high-level action to history
        event = {
            'action_type': 'high_level',
            'action': high_level_action,
            'complexity': 0.0,
            'node_count_reduction': 0.0
        }
        if temperature is not None:
            event['target_probability'] = target_high_level_action_prob if self.training else target_high_level_action_prob.item()
            event['behavior_probability'] = high_level_action_prob.detach() if self.training else high_level_action_prob.item()
        else:
            event['probability'] = high_level_action_prob if self.training else high_level_action_prob.item()
        
        return event, high_level_action
    
    def apply_internal_perceptron(self, features: torch.Tensor, temperature: Optional[float] = None) -> Tuple[dict, int]:
        internal_logits = self.internal_actor(features)
        if temperature is not None:
            target_probs = F.softmax(internal_logits, dim=-1)
            internal_probs = F.softmax(internal_logits/temperature, dim=-1)
        else:
            internal_probs = F.softmax(internal_logits, dim=-1)
        internal_action = torch.multinomial(internal_probs, 1).item()
        internal_action_prob = internal_probs[:,internal_action]
        if temperature is not None:
            target_internal_action_prob = target_probs[:,internal_action]

        event = {
            'action_type': 'internal',
            'action': internal_action,
            'node_count_reduction': 0.0
        }
        if temperature is not None:
            event['target_probability'] = target_internal_action_prob if self.training else target_internal_action_prob.item()
            event['behavior_probability'] = internal_action_prob.detach() if self.training else internal_action_prob.item()
        else:
            event['probability'] = internal_action_prob if self.training else internal_action_prob.item()
        
        return event, internal_action
    
    def apply_external_perceptron(self, features: torch.Tensor, validity_mask: torch.Tensor, temperature: Optional[float] = None) -> Tuple[dict, int, torch.Tensor]:
        # valid_actor_weights = self.actor.weight * validity_mask.unsqueeze(1)
        # valid_actor_bias = self.actor.bias * validity_mask
        # logits = validity_mask.log() + F.linear(features, valid_actor_weights, valid_actor_bias)
        logits = validity_mask.log() + self.actor(features)
        if temperature is not None:
            target_probs = F.softmax(logits, dim=-1)
            action_probs = F.softmax(logits/temperature, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            action_prob = action_probs[:,action]
            target_action_prob = target_probs[:,action]
        else:
            action_probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            action_prob = action_probs[:,action]
        
        event = {
            'action_type': 'external',
            'action': action,
            'complexity': 0.0,
        }
        if temperature is not None:
            event['target_probability'] = target_action_prob if self.training else target_action_prob.item()
            event['behavior_probability'] = action_prob.detach() if self.training else action_prob.item()
        else:
            event['probability'] = action_prob if self.training else action_prob.item()
        
        return event, action, action_probs
    
    def apply_teleport_ffn(self, features: torch.Tensor, temperature: Optional[float] = None) -> torch.Tensor:
        teleport_logits = self.teleport_ffn(features).transpose(-1,-2)
        if temperature is not None:
            target_probs = F.softmax(teleport_logits, dim=-1)
            teleport_probs = F.softmax(teleport_logits/temperature, dim=-1)
        else:
            teleport_probs = F.softmax(teleport_logits, dim=-1)
        teleport_action = torch.multinomial(teleport_probs, 1).item()
        teleport_action_prob = teleport_probs[:,teleport_action]
        if temperature is not None:
            target_teleport_action_prob = target_probs[:,teleport_action]
        event = {
            'action_type': 'teleport',
            'action': teleport_action,
            'complexity': 0.0,
            'node_count_reduction': 0.0
        }
        if temperature is not None:
            event['target_probability'] = target_teleport_action_prob if self.training else target_teleport_action_prob.item()
            event['behavior_probability'] = teleport_action_prob.detach() if self.training else teleport_action_prob.item()
        else:
            event['probability'] = teleport_action_prob if self.training else teleport_action_prob.item()
        
        return event, teleport_action, teleport_probs

    def apply_teleport(self, state: SympleState, **kwargs) -> Tuple[dict, tuple[int, ...]]:
        coords_and_nodes = state.en.get_coords_and_nodes()
        combined_hidden_list = [torch.cat([state.h_glob[-1], node.hidden], dim=-1) for _, node in coords_and_nodes]
        combined_hidden = torch.cat(combined_hidden_list, dim=0)
        
        event, teleport_action, teleport_probs = self.apply_teleport_ffn(combined_hidden, **kwargs)
        
        return event, teleport_action, teleport_probs
    
    def policy(
            self,
            state: SympleState,
            validity_mask: torch.Tensor,
            recursion_depth: int = 5,
            temperature: Optional[float] = None,
            high_level_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[Dict], int, torch.Tensor, SympleState]:
        history = []

        current_node = state.en.get_node(state.coord)
        features = torch.cat([current_node.hidden, state.h_glob[-1]], dim=-1)

        # Decide whether to act, further process the expression, teleport, or finish
        event, high_level_action = self.apply_high_level_perceptron(features, recursion_depth, temperature = temperature, high_level_mask=high_level_mask)
        history.append(event)

        if high_level_action == 0:
            # Internal op
            event, internal_action = self.apply_internal_perceptron(features, temperature = temperature)

            current_node, complexity, state.h_glob, state.c_glob = self.apply_internal_op(current_node, internal_action, state.h_glob, state.c_glob)

            event['complexity'] = complexity

            history.append(event)
            
            sub_history, action, action_probs, state = self.policy(
                state,
                validity_mask,
                recursion_depth - 1,
                temperature,
                high_level_mask
            )
            history.extend(sub_history)
            return history, action, action_probs, state
        elif high_level_action == 1:
            # Teleport
            event, teleport_action, teleport_probs = self.apply_teleport(state, temperature = temperature)
            history.append(event)
            return history, teleport_action, teleport_probs, state
        elif high_level_action == 2:
            # External op
            event, action, action_probs = self.apply_external_perceptron(features, validity_mask, temperature = temperature)
            history.append(event)
            return history, action, action_probs, state
        else:
            # Finish
            event = {
                'action_type': 'finish',
                'action': None,
                'complexity': 0.0,
                'node_count_reduction': 0.0
            }
            history.append(event)
            return history, None, None, state

    def step(self, state: SympleState, env: Symple, **policy_kwargs):
        done = False
        current_node = state.en.get_node(state.coord)
        history, action, _, state = self.policy(
            state,
            env.get_validity_mask(current_node),
            **policy_kwargs
        )
        
        # Add reward and coordinates to all action history
        for entry in history:
            entry['reward'] = env.time_penalty - env.compute_penalty_coefficient * entry.get('complexity', 0)
            entry['coordinates'] = state.coord

        if history[-1]['action_type'] == 'teleport':
            state.coord = state.en.get_coords()[action]
        elif history[-1]['action_type'] == 'finish':
            done = True
        else:
            state.en, state.coord, reward, node_count_reduction = env.step(state.en, state.coord, action)
            history[-1]['reward'] = reward
            history[-1]['node_count_reduction'] = node_count_reduction
        
        return state, done, history

    def forward(self, state: ExprNode, env: Symple,
                behavior_policy: Optional[
                    Union[Callable[
                        [ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]
                    ], Tuple[str, float]]
                ] = None,
                min_steps: int = 0,
                max_steps: int = inf
                ) -> Union[
                    Tuple[List[Dict], ExprNode],
                    ExprNode
                ]:
        if behavior_policy:
            return self.off_policy_forward(state, env, behavior_policy, min_steps, max_steps)
        
        h_glob = torch.zeros((self.lstm.num_layers, 1, self.global_hidden_size), device=self.device, dtype=DEFAULT_DTYPE)
        c_glob = torch.zeros((self.lstm.num_layers, 1, self.global_hidden_size), device=self.device, dtype=DEFAULT_DTYPE)
        
        # Apply to all nodes in the tree
        state = self.apply_binary_lstm(state)
        
        symple_state = SympleState(state, (), h_glob, c_glob)
        done = False
        history = []
        steps = 0

        while steps <= max_steps and not done:

            high_level_mask = torch.tensor(
                ([[1,1,1,1]] if steps >= min_steps else [[1,1,1,0]]),
                device = self.device,
                dtype = DEFAULT_DTYPE
            ) # Mask out finish action
            symple_state, done, step_history = self.step(
                symple_state,
                env,
                high_level_mask = high_level_mask,
            )
            history.extend(step_history)
            steps += 1


        return history, symple_state.en.reset_tensors()
    
    def off_policy_step(self, state: SympleState, env: Symple,
                        behavior_policy: Union[
                            Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]],
                        Tuple[str, float]
                        ],
                        **policy_kwargs
                        ) -> Tuple[SympleState, bool, Dict]:

        if isinstance(behavior_policy, tuple):
            if behavior_policy[0] == 'temperature':
                temperature = behavior_policy[1]
                return self.step(state, env, temperature=temperature)
            else:
                raise ValueError(f"Invalid behavior policy type: {behavior_policy[0]}")

        current_node = state.en.get_node(state.coord)
        validity_mask = env.get_validity_mask(current_node)
        history, action, target_probs, state = self.policy(
            state,
            validity_mask,
            temperature=1.0,
            **policy_kwargs
        )

        for entry in history:
            entry['reward'] = env.time_penalty - (env.compute_penalty_coefficient * entry.get('complexity', 0))
            entry['coordinates'] = state.coord

        done = False
        if history[-1]['action_type'] == 'teleport':
            state.coord = action
        elif history[-1]['action_type'] == 'finish':
            done = True
        else:
            behavior_probs = behavior_policy(state.en, validity_mask)
            action = torch.multinomial(torch.tensor(behavior_probs), 1).item()

            target_action_prob = target_probs[:,action]
            behavior_action_prob = behavior_probs[:,action]
            state.en, state.coord, reward, node_count_reduction = env.step(state.en, state.coord, action)
            
            event = {
                'action_type': 'external',
                'action': action,
                'target_probability': target_action_prob if self.training else target_action_prob.item(),
                'behavior_probability': behavior_action_prob.detach() if self.training else behavior_action_prob.item(),
                'reward': reward,
                'complexity': 0.0,
                'node_count_reduction': node_count_reduction,
                'coordinates': state.coord
            }
            history.append(event)

        return state, done, history
    
    def off_policy_forward(self, state: ExprNode, env: Symple,
                           behavior_policy: Union[Callable[[ExprNode, tuple[int, ...], Symple], Union[torch.Tensor, List[float]]], Tuple[str, float]],
                           min_steps: int = 0,
                           max_steps: int = inf
                           ) -> Tuple[List[Dict], ExprNode]:
        
        h_glob = torch.zeros((self.lstm.num_layers, 1, self.global_hidden_size), device=self.device, dtype=DEFAULT_DTYPE)
        c_glob = torch.zeros((self.lstm.num_layers, 1, self.global_hidden_size), device=self.device, dtype=DEFAULT_DTYPE)
        
        
        # Apply to all nodes in the tree
        state = self.apply_binary_lstm(state)
        done = False
        symple_state = SympleState(state, (), h_glob, c_glob)
        history = []
        steps = 0
        
        while steps <= max_steps and not done:
            high_level_mask= torch.tensor(
                ([[1,0,1,1]] if steps >= min_steps else [[1,0,1,0]]),
                device = self.device,
                dtype = DEFAULT_DTYPE
            ) # Mask out teleport action
            symple_state, done, step_history = self.off_policy_step(
                symple_state,
                env,
                behavior_policy,
                high_level_mask = high_level_mask
            )
            history.extend(step_history)
            steps += 1

        
        return history, symple_state.en.reset_tensors()

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        self.device = device
        return super(SympleAgent, self).to(device, dtype)