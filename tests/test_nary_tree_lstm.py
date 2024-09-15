import pytest
import torch

from symple.model.nary_tree_lstm import NaryTreeLSTM


@pytest.fixture
def simple_binary_tree():
    # Create a simple binary tree with 3 nodes
    #     1
    #    / \
    #   2   3
    nodes = torch.tensor([[2.0], [3.0], [1.0]])
    children = torch.tensor([
        [3, 3],  # Leaf node (2) has no children
        [3, 3],  # Leaf node (3) has no children
        [0, 1],  # Root node (1) has children 2 and 3
    ])
    return nodes, children

def test_nary_tree_lstm_forward_pass(simple_binary_tree):
    nodes, children = simple_binary_tree
    N, input_size, hidden_size = 2, 1, 4
    
    model = NaryTreeLSTM(N, input_size, hidden_size)
    output = model(nodes, children)
    
    assert output.shape == (3, hidden_size)
    assert torch.all(output[0] != 0)  # Root node should have non-zero output
    assert torch.all(output[1] != 0)  # Left child should have non-zero output
    assert torch.all(output[2] != 0)  # Right child should have non-zero output

def test_nary_tree_lstm_pattern_recognition(simple_binary_tree):
    nodes, children = simple_binary_tree
    N, input_size, hidden_size = 2, 1, 4
    
    model = NaryTreeLSTM(N, input_size, hidden_size)
    
    # Define a simple pattern: root > left child and root < right child
    def pattern_recognition(output):
        root, left, right = output[0], output[1], output[2]
        return torch.all(root > left) and torch.all(root < right)
    
    # Train the model to recognize the pattern
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        loss = torch.tensor(0.0, requires_grad=True)
        optimizer.zero_grad()
        output = model(nodes, children)
        loss = (torch.relu(output[1] - output[0]) ** 2).sum()
        loss += (torch.relu(output[0] - output[2]) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    # Test if the model learned the pattern
    with torch.no_grad():
        output = model(nodes, children)
        assert pattern_recognition(output).item(), "Model failed to learn the simple pattern"

