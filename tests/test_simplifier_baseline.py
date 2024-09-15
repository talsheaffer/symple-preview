import matplotlib.pyplot as plt
import pytest
import torch
from torch import optim

from symple.expr import ExprNode
from symple.expr.actions import ACTIONS
from symple.model.simplifier_baseline import SimplifierBaseline
from symple.training.policy_gradient import AgentEnv, policy_gradient_loss

# Sample data for testing
batch = [
    "(1 + 1) * x - x",
    "(1 + x) * 1",
    "x * x - x * (1 * x)",
    "(1 * 1 * x * 1) - 1 * x",
]

@pytest.fixture
def model():
    return SimplifierBaseline(reader_hidden_size=10, policy_hidden_size=10, policy_layers=2, n_actions=len(ACTIONS))

def test_simplifier_baseline(model):
    torch.manual_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exprs_nodes = [ExprNode.from_sympy_str(expr) for expr in batch]

    R_avg_init = None
    
    R_avg_rec = []
    for _ in range(250):
        J_cum = 0
        R_cum = 0
        optimizer.zero_grad()
        for expr_node in exprs_nodes:
            J, R = policy_gradient_loss(expr_node, model, steps=10, discount_factor=0.8)
            J_cum -= J
            R_cum += R
        
        R_avg = R_cum / len(exprs_nodes)
        R_avg_rec.append(R_avg)
        
        if R_avg_init is None:
            R_avg_init = R_avg
        
        J_cum.backward()
        optimizer.step()        

    plt.plot(R_avg_rec)
    plt.show()

    env = AgentEnv(model)
    expr = exprs_nodes[0].clone()

    assert R_avg < R_avg_init, "Loss did not decrease"
