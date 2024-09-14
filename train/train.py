
import time
import matplotlib.pyplot as plt
import sys


import sympy as sp

import pandas as pd
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

for epoch in range(num_epochs):
    # Shuffle the dataset
    shuffled_data = df['expr'].sample(frac=1).reset_index(drop=True)
    
    for i in range(0, batch_size * (len(shuffled_data) // batch_size), batch_size):
        batch = shuffled_data[i:i+batch_size].apply(ExprNode.from_sympy).tolist()
        
        # Measure time for training on the batch
        start_time = time.time()
        avg_return = train_on_batch(agent, batch, optimizer,
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
        
        batch_number = epoch * (len(shuffled_data) // batch_size) + (i // batch_size) + 1
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_number}, Return: {avg_return:.4f}, Batch Time: {batch_time:.4f} seconds, Avg Eval Time: {avg_eval_time:.4f} seconds")

avg_time_per_batch = total_time / (num_epochs * (len(df) // batch_size))
print(f"Training completed. Average time per batch: {avg_time_per_batch:.4f} seconds")

# Plot the return history
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(returns) + 1), returns)
plt.title('Return vs Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Return')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Add horizontal line at y=0
plt.savefig(ROOT_DIR + '/train/return_history.png')

# Plot the evaluation time history
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(eval_times) + 1), eval_times)
plt.title('Average Evaluation Time per Expression vs Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Average Evaluation Time (seconds)')
plt.savefig(ROOT_DIR + '/train/eval_time_history.png')


# Plot both return and evaluation time on the same graph
plt.figure(figsize=(12, 6))

# Plot return
plt.plot(range(1, len(returns) + 1), returns, label='Return', color='blue')
plt.ylabel('Return', color='blue')
plt.tick_params(axis='y', labelcolor='blue')

# Create a twin axis for evaluation time
ax2 = plt.twinx()
ax2.plot(range(1, len(eval_times) + 1), eval_times, label='Avg Eval Time', color='red')
ax2.set_ylabel('Average Evaluation Time (seconds)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Return and Average Evaluation Time vs Batch Number')
plt.xlabel('Batch Number')

# Add legend
lines1, labels1 = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Add horizontal line at y=0
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig(ROOT_DIR + '/train/return_and_eval_time_history.png')
