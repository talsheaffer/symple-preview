import time
import sys
import os

import sympy as sp

import pandas as pd
import json
from datetime import datetime

import torch

# from src.model.environment import Symple
from src.model.model import SympleAgent
from src.model.training import train_on_batch
from src.model.tree import ExprNode

# from aux_policies import random_policy
# behavior_policy = random_policy

# Load the dataset
from definitions import ROOT_DIR
sys.setrecursionlimit(10000)  # Increase as needed

# Apply from_sympy to df['expr']



with open(ROOT_DIR + "/data/dataset.json", "r") as f:
    df = pd.read_json(f)
df[df.columns[:3]] = df[df.columns[:3]].map(sp.sympify)

# df['symple_envs'] = df['expr'].apply(Symple.from_sympy,
#                                      time_penalty=0., # Additional keyward args to be passed to Symple.from_sympy
#                                      )





# Initialize the agent and optimizer
hidden_size = 128
# embedding_size = 16
agent = SympleAgent(
    hidden_size,
    ffn_n_layers=2,
    # lstm_n_layers=2,
)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 32


total_time = 0
returns = []
total_time = 0
eval_times = []


# Generate a unique filename for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
training_data_dir = os.path.join(ROOT_DIR, "train", "training_data")
os.makedirs(training_data_dir, exist_ok=True)
json_filename = os.path.join(training_data_dir, f"training_data_{timestamp}.json")

training_data = []

for epoch in range(1, num_epochs + 1):
    # Shuffle the dataset
    shuffled_data = df['expr'].sample(frac=1).reset_index(drop=True)
    n_batches = len(shuffled_data) // batch_size
    
    for i in range(0, batch_size * n_batches, batch_size):
        batch = shuffled_data[i:i+batch_size].apply(ExprNode.from_sympy).tolist()
        
        # Measure time for training on the batch
        start_time = time.time()
        avg_return, batch_history = train_on_batch(agent, batch, optimizer,
                                    # behavior_policy=behavior_policy,
                                    **dict(
                                        time_penalty=0.02,
                                        min_steps=20,
                                    )
                                )
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        
        returns.append(avg_return)
        avg_eval_time = batch_time / batch_size  # Calculate average evaluation time per expression
        eval_times.append(avg_eval_time)
        avg_ncr = sum([sum([step['node_count_reduction'] for step in history]) for history in batch_history]) / len(batch_history)
        
        batch_number = (i // batch_size) + 1
        print(f"Epoch {epoch}/{num_epochs}, Batch {batch_number}/{n_batches}, Return: {avg_return:.4f}, NCR: {avg_ncr:.4f}, Batch Time: {batch_time:.4f} seconds, Avg Eval Time: {avg_eval_time:.4f} seconds")

        # Save batch data
        batch_data = {
            'epoch': epoch,
            'batch_number': batch_number,
            'avg_return': avg_return,
            'avg_ncr': avg_ncr,
            'batch_time': batch_time,
            'avg_eval_time': avg_eval_time,
            'history': batch_history
        }
        training_data.append(batch_data)

    # Save data after each epoch
    with open(json_filename, 'w') as f:
        json.dump(training_data, f, indent=2)

avg_time_per_batch = total_time / (num_epochs * (len(df) // batch_size))
print(f"Training completed. Average time per batch: {avg_time_per_batch:.4f} seconds")
print(f"Training data saved to: {json_filename}")


