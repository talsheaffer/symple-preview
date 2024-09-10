
import time
import matplotlib.pyplot as plt
import sys


import sympy as sp

import pandas as pd
import torch

from src.model.environment import Symple
from src.model.model import SympleAgent
from src.model.training import train_on_batch

# Load the dataset
from definitions import ROOT_DIR
sys.setrecursionlimit(10000)  # Increase as needed

# Apply from_sympy to df['expr']



with open(ROOT_DIR + "/data/dataset.json", "r") as f:
    df = pd.read_json(f)
df[df.columns[:3]] = df[df.columns[:3]].map(sp.sympify)

df['symple_envs'] = df['expr'].apply(Symple.from_sympy,
                                     time_penalty=0., # Additional keyward args to be passed to Symple.from_sympy
                                     )





# Initialize the agent and optimizer
hidden_size = 64
embedding_size = 32
agent = SympleAgent(
    hidden_size, embedding_size,

)
optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

# Training loop
num_epochs = 5
batch_size = 32


total_time = 0

losses = []
total_time = 0

for epoch in range(num_epochs):
    # Shuffle the dataset
    shuffled_data = df['symple_envs'].sample(frac=1).reset_index(drop=True)
    
    for i in range(0, len(shuffled_data), batch_size):
        batch = shuffled_data[i:i+batch_size].tolist()
        
        # Measure time for training on the batch
        start_time = time.time()
        avg_loss = train_on_batch(agent, batch, optimizer)
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        
        losses.append(avg_loss)
        
        batch_number = epoch * (len(shuffled_data) // batch_size) + (i // batch_size) + 1
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_number}, Loss: {avg_loss:.4f}, Batch Time: {batch_time:.4f} seconds")

avg_time_per_batch = total_time / (num_epochs * (len(df) // batch_size))
print(f"Training completed. Average time per batch: {avg_time_per_batch:.4f} seconds")

# Plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(losses) + 1), losses)
plt.title('Loss vs Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()

