import torch
import torch.nn.functional as F
from torch import nn

from src.model.environment import NUM_OPS, Symple
from src.model.state import SympleState
from src.model.ffn import FFN
from src.model.tree import VOCAB_SIZE, ExprNode
from src.model.nary_tree_lstm import NaryTreeLSTM

from src.model.model_default import DEFAULT_DEVICE, DEFAULT_DTYPE

from collections import OrderedDict

from typing import Callable, Union, List, Tuple, Optional, Dict
from sympy import Expr
from numpy import inf

def compose(f, g):
    return lambda x: f(g(x))

def with_kwargs(f, **kwargs):
    return lambda *args: f(*args, **kwargs)

NUM_INTERNAL_OPS = 5
INTERNAL_OPS = [
    (lambda agent: agent.apply_ffn, lambda agent: (lambda state: agent.ffn_complexity), 'ffn'),
    (
        lambda agent: compose(agent.apply_global_lstm, agent.apply_ffn),
        lambda agent: (lambda state: agent.ffn_complexity + agent.glstm_complexity),
        'ffn - glstm'
    ),
    (lambda agent: agent.apply_global_lstm, lambda agent: (lambda state: agent.glstm_complexity), 'glstm')
    ]
for i in range(NUM_INTERNAL_OPS-3):
    depth = i//2
    if i %2 == 0:
        INTERNAL_OPS.append(
            (
                lambda agent: with_kwargs(agent.apply_ternary_lstm, depth = depth),
                lambda agent: (lambda state: agent.tlstm_complexity * state.current_node.node_count(depth=depth)),
                f'tlstm (depth={depth})'
            )
        )
    else:
        INTERNAL_OPS.append(
            (
            lambda agent: compose(agent.apply_global_lstm, with_kwargs(agent.apply_ternary_lstm, depth = depth)),
            lambda agent: (lambda state: agent.tlstm_complexity * state.current_node.node_count(depth=depth) + agent.glstm_complexity),
            f'tlstm (depth={depth}) - glstm'
            )
        )


class SympleAgent(nn.Module):
    """
    Currently only actor, no critic.
    """

    def __init__(
        self,
        hidden_size: int,
        global_hidden_size: Optional[int] = None,
        ffn_n_layers: int = 1,
        lstm_n_layers: int = 1,
    ):
        super(SympleAgent, self).__init__()
        self.hidden_size = hidden_size
        self.global_hidden_size = global_hidden_size if global_hidden_size is not None else hidden_size
        self.vocab_size = VOCAB_SIZE
        self.num_ops = NUM_OPS
        self.num_internal_ops = NUM_INTERNAL_OPS
        self.feature_size = self.hidden_size + self.global_hidden_size + 8
        self.global_lstm_input_size = self.hidden_size + 8

        # nn modules
        self.blstm = NaryTreeLSTM(2, self.vocab_size, self.hidden_size)
        self.tlstm = NaryTreeLSTM(3, self.vocab_size, self.hidden_size)
        self.lstm = nn.LSTM(self.global_lstm_input_size, self.global_hidden_size, num_layers=lstm_n_layers, batch_first=True)
        self.ffn = FFN(self.hidden_size, self.hidden_size, self.hidden_size, n_layers=ffn_n_layers)

        # FFNs for high-level and internal decisions
        self.high_level_actor = FFN(self.feature_size, self.hidden_size, 4, n_layers=ffn_n_layers)  # Changed to 4 for teleport and finish actions
        self.teleport_ffn = FFN(self.feature_size, self.hidden_size, 1, n_layers=ffn_n_layers)
        self.internal_actor = FFN(self.feature_size, self.hidden_size, self.num_internal_ops, n_layers=ffn_n_layers)
        self.actor = FFN(self.feature_size, self.hidden_size, self.num_ops, n_layers=ffn_n_layers)

        # Value function estimator
        self.value_function = FFN(self.feature_size, self.hidden_size, 1, n_layers=ffn_n_layers)

        # internal ops and their compute complexity. Including certain compositions of elementary internal ops
        self.ffn_complexity = self.hidden_size**2 * self.ffn.n_layers
        self.teleport_ffn_complexity =  self.hidden_size * (self.feature_size + self.hidden_size * self.teleport_ffn.n_layers + 1)
        self.glstm_complexity = 4 * (self.global_hidden_size * self.global_lstm_input_size * self.lstm.num_layers)
        self.blstm_complexity = 4 * (self.hidden_size * (self.vocab_size + self.hidden_size) * 2) 
        self.tlstm_complexity = 4 * (self.hidden_size * (self.vocab_size + self.hidden_size) * 3)
       
        self.internal_ops = [(op(self), complexity_func(self)) for op, complexity_func, _ in INTERNAL_OPS]
        
        self.high_level_op_indices = OrderedDict(
            internal = 0,
            teleport = 1,
            external = 2,
            finish = 3
        )
        self.high_level_op_labels = list(self.high_level_op_indices.keys())
        
        # auxiliary variables
        self.device = DEFAULT_DEVICE
        
    @property
    def tz(self) -> torch.Tensor:
        """
        Returns a tensor of zeros with shape (1, hidden_size).

        Returns:
            torch.Tensor: A tensor of zeros.
        """
        return torch.zeros((1, self.hidden_size), device=self.device, dtype=DEFAULT_DTYPE)

    def apply_internal_op(self, state: SympleState, op_num: int) -> Tuple[SympleState, float]:
        """
        Applies an internal operation to the input expression tree recursively.

        Args:
            state (SympleState): The current state.
            op_num (int): The number of the internal operation to apply.

        Returns:
            Tuple[SympleState, float]: The updated state and the complexity of the operation.
        """
        op, complexity_func = self.internal_ops[op_num]
        complexity = complexity_func(state)
        state = op(state)
        return state, complexity
    
    def apply_global_lstm(self, state: SympleState) -> SympleState:
        """
        Applies a global LSTM to the input expression tree recursively.

        Args:
            state (SympleState): The current state.

        Returns:
            SympleState: The updated state.
        """
        current_node = state.current_node
        lstm_input = torch.cat([current_node.hidden, state.state_tensor], dim=-1)
        _, (state.h_glob, state.c_glob) = self.lstm(lstm_input.unsqueeze(0), (state.h_glob, state.c_glob))
        return state
    
    def apply_ffn(self, state: SympleState) -> SympleState:
        """
        Applies the feedforward network to the input expression node.

        Args:
            state (SympleState): The current state.

        Returns:
            SympleState: The updated state.
        """
        current_node = state.current_node
        current_node.hidden = self.ffn(current_node.hidden)
        state.substitute_current_node(current_node)
        return state
    
    def apply_teleport_ffn(self, state: SympleState) -> SympleState:
        """
        Applies the teleport feedforward network to the input expression tree recursively.

        Args:
            state (SympleState): The current state.

        Returns:
            SympleState: The updated state.
        """
        coords_and_nodes = state.en.get_coords_and_nodes()
        n_nodes = len(coords_and_nodes)
        h_glob = state.h_glob[-1].repeat(n_nodes, 1)
        state_tensor = state.state_tensor.repeat(n_nodes, 1)
        combined_hidden_list = [node.hidden for _, node in coords_and_nodes]
        combined_hidden = torch.cat(combined_hidden_list, dim=0)
        features = torch.cat([h_glob, combined_hidden, state_tensor], dim=-1)
        teleport_logits = self.teleport_ffn(features).transpose(-1, -2)
        return teleport_logits
    
    def apply_binary_lstm(self, input: ExprNode, depth: int = inf) -> ExprNode:
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

        if input.hidden is not None:
            return input
        
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
    
    def apply_ternary_lstm_to_node(self, input: ExprNode, depth: int = 0) -> ExprNode:
        """
        Applies a ternary LSTM to the input expression tree recursively.

        Args:
            input (ExprNode): The root node of the expression tree.
            depth (int, optional): The maximum depth to traverse. Defaults to 0.

        Returns:
            ExprNode: The input node with updated hidden and cell states.
        """
        if depth > 0:
            input.a = self.apply_ternary_lstm_to_node(input.a, depth=depth - 1) if input.a is not None else None
            input.b = self.apply_ternary_lstm_to_node(input.b, depth=depth - 1) if input.b is not None else None
        input.ensure_parenthood()

        (input.hidden, input.cell) = self.tlstm(
            input.arg_hot,
            (
                input.hidden if input.hidden is not None else self.tz,
                input.a.hidden if input.a is not None else self.tz,
                input.b.hidden if input.b is not None else self.tz,
            ),
            (
                input.cell if input.hidden is not None else self.tz,
                input.a.cell if input.a is not None else self.tz,
                input.b.cell if input.b is not None else self.tz,
            )
        )
        
        return input

    def apply_ternary_lstm(self, state: SympleState, depth: int = 0) -> SympleState:
        """
        Applies a ternary LSTM to the current node in the state.

        Args:
            state (SympleState): The current state.
            depth (int, optional): The maximum depth to traverse. Defaults to 0.

        Returns:
            SympleState: The updated state.
        """
        current_node = state.current_node
        current_node = self.apply_ternary_lstm_to_node(current_node, depth)
        state.substitute_current_node(current_node)
        return state
    
    def get_policy_logits(self, state: SympleState, validity_mask: torch.Tensor, high_level_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the policy logits for the current state.

        Args:
            state (SympleState): The current state.
            validity_mask (torch.Tensor): Mask for valid actions.
            high_level_mask (Optional[torch.Tensor], optional): Mask for high-level actions. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Logits for high-level, internal, external, and teleport actions.
        """
        current_node = state.en.get_node(state.coord)
        features = torch.cat([current_node.hidden, state.h_glob[-1], state.state_tensor], dim=-1)

        l_high = self.high_level_actor(features)
        if high_level_mask is not None:
            l_high = l_high + high_level_mask.log()

        l_internal = self.internal_actor(features)
        l_ext = self.actor(features) + validity_mask.log()

        l_teleport = self.apply_teleport_ffn(state)

        return l_high, l_internal, l_ext, l_teleport
    
    def policy(self, state: SympleState, validity_mask: torch.Tensor, high_level_mask: Optional[torch.Tensor] = None, temperature: float = 1.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the policy probabilities for the current state.

        Args:
            state (SympleState): The current state.
            validity_mask (torch.Tensor): Mask for valid actions.
            high_level_mask (Optional[torch.Tensor], optional): Mask for high-level actions. Defaults to None.
            temperature (float, optional): Temperature for softmax. Defaults to 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Probabilities for high-level, internal, external, and teleport actions.
        """
        l_high, l_internal, l_ext, l_teleport = self.get_policy_logits(state, validity_mask, high_level_mask)
        p_high = F.softmax(l_high / temperature, dim=-1)
        p_internal = F.softmax(l_internal / temperature, dim=-1)
        p_ext = F.softmax(l_ext / temperature, dim=-1)
        p_teleport = F.softmax(l_teleport / temperature, dim=-1)

        return p_high, p_internal, p_ext, p_teleport

    def estimate_value(self, state: SympleState) -> torch.Tensor:
        """
        Estimates the value of the current state.

        Args:
            state (SympleState): The current state.

        Returns:
            torch.Tensor: The estimated value of the state.
        """
        current_node = state.current_node
        features = torch.cat([current_node.hidden, state.h_glob[-1], state.state_tensor], dim=-1)
        value = self.value_function(features)
        return value

    def step(self, state: SympleState, env: Symple, can_finish: bool = True, temperature: float = 1.) -> Tuple[SympleState, bool, Dict]:
        """
        Performs a single step in the environment.

        Args:
            state (SympleState): The current state.
            env (Symple): The environment.
            can_finish (bool, optional): Whether the agent can finish. Defaults to True.
            temperature (float, optional): Temperature for softmax. Defaults to 1.

        Returns:
            Tuple[SympleState, bool, Dict]: The new state, whether the episode is done, and event information.
        """
        current_node = state.en.get_node(state.coord)
        features = torch.cat([current_node.hidden, state.h_glob[-1], state.state_tensor], dim=-1)

        # Estimate value before taking action
        value = self.estimate_value(state)

        high_level_mask = torch.ones((1, len(self.high_level_op_indices)), device=self.device, dtype=DEFAULT_DTYPE)
        if not can_finish:
            high_level_mask[0, self.high_level_op_indices['finish']] = 0

        l_high = self.high_level_actor(features) + high_level_mask.log()

        high_level_probs = F.softmax(l_high / temperature, dim=-1)

        high_level_action = torch.multinomial(high_level_probs, 1).item()
        high_level_action_prob = high_level_probs[:, high_level_action]
        high_level_action = self.high_level_op_labels[high_level_action]

        done = False

        if high_level_action == 'internal':
            l_internal = self.internal_actor(features)
            internal_probs = F.softmax(l_internal / temperature, dim=-1)
            action = torch.multinomial(internal_probs, 1).item()
            internal_action_prob = internal_probs[:, action]

            prob = internal_action_prob * high_level_action_prob

            state, complexity = self.apply_internal_op(state, action)
            reward = env.time_penalty - env.compute_penalty_coefficient * complexity
            node_count_reduction = 0

        elif high_level_action == 'teleport':
            l_teleport = self.apply_teleport_ffn(state)

            teleport_probs = F.softmax(l_teleport / temperature, dim=-1)
            action = torch.multinomial(teleport_probs, 1).item()
            teleport_action_prob = teleport_probs[:, action]

            state.coord = state.en.get_coords()[action]

            prob = teleport_action_prob * high_level_action_prob

            reward = env.time_penalty
            node_count_reduction = 0
            complexity = state.en.node_count() * self.teleport_ffn_complexity

        elif high_level_action == 'external':
            validity_mask = env.get_validity_mask(state)
            l_ext = self.actor(features) + validity_mask.log()

            action_probs = F.softmax(l_ext / temperature, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            action_prob = action_probs[:, action]

            state, reward, node_count_reduction = env.step(state, action)

            prob = action_prob * high_level_action_prob

            complexity = 0.0

        elif high_level_action == 'finish':
            done = True
            reward = 0
            node_count_reduction = 0
            action = 0
            complexity = 0.0

            prob = high_level_action_prob
        else:
            raise ValueError(f"Invalid high-level action: {high_level_action}")
        
        event = {
            'action_type': high_level_action,
            'action': action,
            'probability': prob if self.training else prob.item(),
            'complexity': complexity,
            'coordinates': state.coord,
            'reward': reward,
            'node_count_reduction': node_count_reduction,
            'value': value if self.training else value.item()
        }
        return state, done, event

    def off_policy_step_from_probs(self, state: SympleState, env: Symple, target_probs: Dict[str, torch.Tensor], behavior_probs: Dict[str, torch.Tensor]) -> Tuple[SympleState, bool, Dict]:
        """
        Performs an off-policy step using provided probabilities.

        Args:
            state (SympleState): The current state.
            env (Symple): The environment.
            target_probs (Dict[str, torch.Tensor]): Target policy probabilities.
            behavior_probs (Dict[str, torch.Tensor]): Behavior policy probabilities.

        Returns:
            Tuple[SympleState, bool, Dict]: The new state, whether the episode is done, and event information.
        """
        # Estimate value before taking action
        value = self.estimate_value(state)

        high_level_action = torch.multinomial(behavior_probs['high_level'], 1).item()
        behavior_high_level_action_prob = behavior_probs['high_level'][:, high_level_action]
        target_high_level_action_prob = target_probs['high_level'][:, high_level_action]
        high_level_action = self.high_level_op_labels[high_level_action]    

        done = False

        if high_level_action == 'internal':
            action = torch.multinomial(behavior_probs['internal'], 1).item()
            target_prob = target_probs['internal'][:, action]
            behavior_prob = behavior_probs['internal'][:, action]

            state, complexity = self.apply_internal_op(state, action)

            reward = env.time_penalty - env.compute_penalty_coefficient * complexity
            node_count_reduction = 0
        
        elif high_level_action == 'external':
            action = torch.multinomial(behavior_probs['external'], 1).item()
            target_prob = target_probs['external'][:, action]
            behavior_prob = behavior_probs['external'][:, action]

            state, reward, node_count_reduction = env.step(state, action)
            complexity = 0.0

        elif high_level_action == 'teleport':
            action = torch.multinomial(behavior_probs['teleport'], 1).item()
            target_prob = target_probs['teleport'][:, action]
            behavior_prob = behavior_probs['teleport'][:, action]

            state.coord = state.en.get_coords()[action]

            reward = env.time_penalty
            node_count_reduction = 0
            complexity = 0.0
        
        elif high_level_action == 'finish':
            target_prob = torch.ones(1)
            behavior_prob = torch.ones(1)
            done = True
            reward = 0
            node_count_reduction = 0
            action = 0
            complexity = 0.0

        else:
            raise ValueError(f"Invalid high-level action: {high_level_action}")
        
        target_prob = target_prob * target_high_level_action_prob
        behavior_prob = behavior_prob * behavior_high_level_action_prob

        event = {
            'action_type': high_level_action,
            'action': action,
            'target_probability': target_prob if self.training else target_prob.item(),
            'behavior_probability': behavior_prob.detach() if self.training else behavior_prob.item(),
            'complexity': complexity,
            'coordinates': state.coord,
            'reward': reward,
            'node_count_reduction': node_count_reduction,
            'value': value if self.training else value.item()
        }
        return state, done, event
            

    
    def off_policy_step(self, state: SympleState, env: Symple, behavior_policy: Callable[[ExprNode, tuple[int, ...], Symple], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], can_finish: bool = True) -> Tuple[SympleState, bool, Dict]:
        """
        Performs an off-policy step using a provided behavior policy.

        Args:
            state (SympleState): The current state.
            env (Symple): The environment.
            behavior_policy (Callable[[ExprNode, tuple[int, ...], Symple], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): The behavior policy.
            can_finish (bool, optional): Whether the agent can finish. Defaults to True.

        Returns:
            Tuple[SympleState, bool, Dict]: The new state, whether the episode is done, and event information.
        """
        validity_mask = env.get_validity_mask(state)
        high_level_mask = torch.ones((1, len(self.high_level_op_indices)), device=self.device, dtype=DEFAULT_DTYPE)
        if not can_finish:
            high_level_mask[0, self.high_level_op_indices['finish']] = 0

        target_high_probs, target_internal_probs, target_ext_probs, target_teleport_probs = self.policy(state, validity_mask, high_level_mask)
        target_probs = {
            'high_level': target_high_probs,
            'internal': target_internal_probs,
            'external': target_ext_probs,
            'teleport': target_teleport_probs
        }

        behavior_high_probs, behavior_ext_probs, behavior_teleport_probs = behavior_policy(state, validity_mask)
        
        # Adjust given that behvaior policy has no internal option
        # Must make sure internal option is always 0
        i = self.high_level_op_indices['internal']
        assert i == 0, "Internal option must be first"
        p = target_high_probs[:,i,None]
        behavior_high_probs *= (1-p.item())
        behavior_high_probs = torch.cat(
            (p, behavior_high_probs), dim = -1
        )
        behavior_internal_probs = target_internal_probs

        behavior_probs = {
            'high_level': behavior_high_probs,
            'internal': behavior_internal_probs,
            'external': behavior_ext_probs,
            'teleport': behavior_teleport_probs
        }

        return self.off_policy_step_from_probs(state, env, target_probs, behavior_probs)



    def off_policy_step_with_temp(self, state: SympleState, env: Symple, temperature: float, can_finish: bool = True) -> Tuple[SympleState, bool, Dict]:
        """
        Performs an off-policy step using a temperature parameter.

        Args:
            state (SympleState): The current state.
            env (Symple): The environment.
            temperature (float): The temperature parameter for softmax.
            can_finish (bool, optional): Whether the agent can finish. Defaults to True.

        Returns:
            Tuple[SympleState, bool, Dict]: The new state, whether the episode is done, and event information.
        """
        validity_mask = env.get_validity_mask(state)
        high_level_mask = torch.ones((1, len(self.high_level_op_indices)), device=self.device, dtype=DEFAULT_DTYPE)
        if not can_finish:
            high_level_mask[0, self.high_level_op_indices['finish']] = 0

        l_high, l_internal, l_ext, l_teleport = self.get_policy_logits(state, validity_mask, high_level_mask)

        target_probs = {
            'high_level': F.softmax(l_high, dim=-1),
            'internal': F.softmax(l_internal, dim=-1),
            'external': F.softmax(l_ext, dim=-1),
            'teleport': F.softmax(l_teleport, dim=-1)
        }

        behavior_probs = {
            'high_level': F.softmax(l_high / temperature, dim=-1),
            'internal': F.softmax(l_internal / temperature, dim=-1),
            'external': F.softmax(l_ext / temperature, dim=-1),
            'teleport': F.softmax(l_teleport / temperature, dim=-1)
        }

        return self.off_policy_step_from_probs(state, env, target_probs, behavior_probs)

    def forward(self, expr: Union[SympleState, Expr, str],
                env: Symple = Symple(),
                behavior_policy: Optional[
                    Union[Callable[
                        [
                            ExprNode,
                            tuple[int, ...],
                            Symple
                        ],
                        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                    ], Tuple[str, float]]
                ] = None,
                min_steps: int = 0,
                max_steps: int = inf
                ) -> Union[
                    Tuple[List[Dict], ExprNode],
                    SympleState
                ]:
        """
        Performs a forward pass through the model.

        Args:
            expr (Union[ExprNode, Expr, str]): The input expression.
            env (Symple, optional): The environment. Defaults to Symple().
            behavior_policy (Optional[Union[Callable[[ExprNode, tuple[int, ...], Symple], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Tuple[str, float]]], optional): The behavior policy. Defaults to None.
            min_steps (int, optional): Minimum number of steps. Defaults to 0.
            max_steps (int, optional): Maximum number of steps. Defaults to inf.

        Returns:
            Union[Tuple[List[Dict], ExprNode], ExprNode]: The history of events and the final expression node, or just the final expression node.
        """
        if isinstance(expr, (str, Expr)):
            state = SympleState.from_sympy(expr)
        elif isinstance(expr, SympleState):
            state = expr
        else:
            raise ValueError(f"Invalid input type: {type(expr)}")
        
        # Initialize global LSTM states
        state.h_glob = torch.zeros((self.lstm.num_layers, 1, self.global_hidden_size), device=self.device, dtype=DEFAULT_DTYPE)
        state.c_glob = torch.zeros((self.lstm.num_layers, 1, self.global_hidden_size), device=self.device, dtype=DEFAULT_DTYPE)
        
        if behavior_policy:
            return self.off_policy_forward(state, env, behavior_policy, min_steps, max_steps)
        
        history = []
        steps = 0
        done = False

        while steps <= max_steps and not done:
            # (Re-)apply binary LSTM only to new nodes for which hidden states were not learned
            state.en = self.apply_binary_lstm(state.en)
            state.nc = state.node_count()

            # Take step
            state, done, event = self.step(
                state,
                env,
                can_finish = steps >= min_steps,
            )
            history.append(event)
            steps += 1

        state.finish()
        return history, state
    
    def off_policy_forward(self, state: SympleState, env: Symple,
                           behavior_policy: Union[Callable[[ExprNode, tuple[int, ...], Symple], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Tuple[str, float]],
                           min_steps: int = 0,
                           max_steps: int = inf
                           ) -> Tuple[List[Dict], SympleState]:
        """
        Performs an off-policy forward pass through the model.

        Args:
            en (ExprNode): The input expression node.
            env (Symple): The environment.
            behavior_policy (Union[Callable[[ExprNode, tuple[int, ...], Symple], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Tuple[str, float]]): The behavior policy.
            min_steps (int, optional): Minimum number of steps. Defaults to 0.
            max_steps (int, optional): Maximum number of steps. Defaults to inf.

        Returns:
            Tuple[List[Dict], ExprNode]: The history of events and the final expression node.
        """
        done = False
        history = []
        steps = 0
        
        while steps <= max_steps and not done:
            # Reapply binary LSTM only to new nodes for which hidden states were not learned
            state.en = self.apply_binary_lstm(state.en)
            state.nc = state.node_count()
            
            if isinstance(behavior_policy, tuple):
                if behavior_policy[0] == 'temperature':
                    state, done, event = self.off_policy_step_with_temp(
                        state,
                        env,
                        temperature=behavior_policy[1],
                        can_finish=steps >= min_steps
                    )
            elif behavior_policy == 'random':
                temperature = 0.0
                state, done, event = self.off_policy_step_with_temp(
                    state,
                    env,
                    temperature=temperature,
                    can_finish=steps >= min_steps
                )
            elif isinstance(behavior_policy, Callable):
                state, done, event = self.off_policy_step(
                    state,
                    env,
                    behavior_policy,
                    can_finish=steps >= min_steps
                )
            else:
                raise ValueError(f"Invalid behavior policy: {behavior_policy}")
            history.append(event)
            steps += 1

        state.finish()
        return history, state

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        self.device = device
        return super(SympleAgent, self).to(device, dtype)