import torch
from torch import nn
from torch.distributions import Categorical

from symple.expr import ACTIONS, ExprNode
from symple.model import ExprTensor


class AgentEnv:
    def __init__(self, model):
        self.model = model

    def act(self, expr: ExprNode):
        expr_t = ExprTensor.from_node(expr)
        
        pi = self.model.forward(
            nodes=expr_t.nodes, 
            children=expr_t.children, 
            valid_actions=expr_t.valid_actions
        )
        
        if pi.isnan().any():
            return 0, 0
        
        a = Categorical(pi.flatten()).sample()
        a_node = a // len(ACTIONS)
        a_action = a % len(ACTIONS)
        
        old_node = expr_t.sorted_nodes[a_node]
        new_node = list(ACTIONS.keys())[a_action](old_node)
        reward = old_node.node_count() - new_node.node_count()
        old_node.replace(new_node)
        
        return reward, pi[a_node, a_action].log()



def policy_gradient_loss(expr: ExprNode, model: nn.Module, discount_factor: float = 1.0, steps: int = 10, baseline: float = 0.0):
        J = torch.tensor(0.0)
        
        env = AgentEnv(model)
        expr = expr.clone()
        
        action_logit_rec = []
        reward_rec = []
        
        for _ in range(steps):
            reward_cur, action_logit_cur = env.act(expr)
            if action_logit_cur == 0:
                break
            reward_rec.append(reward_cur)
            action_logit_rec.append(action_logit_cur)

        for reward, action_prob in zip(reversed(reward_rec), reversed(action_logit_rec)):
            J *= discount_factor
            J += (reward - baseline) * action_prob
        
        return J, sum(reward_rec)