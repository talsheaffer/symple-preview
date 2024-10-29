import pytest
from torch import optim

from symple.expr import ExprNode
from symple.expr.actions import ACTIONS
from symple.model.simplifier_baseline import SimplifierBaseline
from symple.training.policy_gradient import policy_gradient_loss

# Sample data for testing
batch = [
    "x**2 + 2*x + 1", 
    "x**2 - 2*x + 1", 
    "x**2 + 4*x + 4",
]

@pytest.fixture
def model():
    return SimplifierBaseline(reader_hidden_size=10, policy_hidden_size=10, policy_layers=2, n_actions=len(ACTIONS))

def test_simplifier_baseline(model):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exprs_nodes = [ExprNode.from_sympy_str(expr) for expr in batch]

    initial_loss = None
    for _ in range(10):  # Train for 10 epochs
        loss = 0
        optimizer.zero_grad()
        for expr_node in exprs_nodes:
            loss += policy_gradient_loss(expr_node, model, steps=3)
        
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss
        
    assert loss < initial_loss, "Loss did not decrease"
