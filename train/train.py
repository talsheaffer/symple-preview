import time
import os

import sympy as sp

import pandas as pd
import json
from datetime import datetime

import torch

# from src.model.environment import Symple
from src.model.model import SympleAgent
from src.model.training import train_on_batch_with_value_function_baseline
from src.model.state import SympleState

from definitions import ROOT_DIR

# Load the dataset
with open(ROOT_DIR + "/data/dataset.json", "r") as f:
    df = pd.read_json(f)
df[df.columns[:3]] = df[df.columns[:3]].map(sp.sympify)

# Generate a unique filename for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
training_data_dir = os.path.join(ROOT_DIR, "train", "training_data")
os.makedirs(training_data_dir, exist_ok=True)
json_filename = os.path.join(training_data_dir, f"training_data_{timestamp}.json")


model_save_dir = os.path.join(ROOT_DIR, 'train', 'models')
os.makedirs(model_save_dir, exist_ok=True)
model_filename = f'model_{timestamp}.pth'
model_path = os.path.join(model_save_dir, model_filename)







agent = SympleAgent(
    hidden_size = 128,
    # global_hidden_size=256,
    ffn_n_layers=2,
    lstm_n_layers=2
)

# # Load the most recent model
# model_files = [f for f in os.listdir(model_save_dir) if f.startswith('model_') and f.endswith('.pth')]
# if model_files:
#     latest_model = max(model_files, key=lambda x: datetime.strptime(x, 'model_%Y%m%d_%H%M%S.pth'))
#     agent.load_state_dict(torch.load(os.path.join(model_save_dir, latest_model)))
#     print(f'Loaded model from: {latest_model}')
# else:
#     print('No previous model found. Starting training from scratch.')
# model_path = os.path.join(model_save_dir, latest_model)



# Define learning rate schedule
initial_lr = 0.001
lr_decay_factor = 0.9

# Initialize Adam optimizer
optimizer = torch.optim.Adam(
    agent.parameters(),
    lr=initial_lr,
    weight_decay = 0.001
)

# Training loop
num_epochs = 30
batch_size = 32


total_time = 0
returns = []
total_time = 0
eval_times = []


training_data = []

# Initialize value function estimate
V = torch.zeros(1, device=agent.device)

overall_batch_num = 0

for epoch in range(1, num_epochs + 1):
    # behavior_policy = ('temperature', 2.0 + .1 * (epoch - 1)) if epoch < 20 else None
    behavior_policy = None
    print(f"Epoch {epoch}/{num_epochs}, Behavior Policy: {behavior_policy}")
    # Update learning rate based on epoch
    if epoch < 30:
        current_lr = initial_lr * (lr_decay_factor ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    # Shuffle the dataset
    shuffled_data = df['expr'].sample(frac=1).reset_index(drop=True)
    n_batches = len(shuffled_data) // batch_size

    epoch_data = []
    
    for i in range(0, batch_size * n_batches, batch_size):
        batch = shuffled_data[i:i+batch_size].apply(SympleState.from_sympy).tolist()
        
        overall_batch_num += 1
        batch_number_in_epoch = (i // batch_size) + 1

        # Measure time for training on the batch
        start_time = time.time()
        avg_return, batch_history, output_expr_nodes, V = train_on_batch_with_value_function_baseline(
            agent,
            batch,
            optimizer,
            V=V,
            batch_num=overall_batch_num,
            behavior_policy=behavior_policy,
            agent_forward_kwargs=dict(
                min_steps=30,
                max_steps=10000,
            )
        )
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        

        # Compute and verify node count reduction
        for input_expr, output_expr, history in zip(
            shuffled_data[i:i+batch_size].tolist(), output_expr_nodes, batch_history
        ):
            history['input'] = str(input_expr)
            history['output'] = str(output_expr.to_sympy())
            history['n_steps'] = len(history['example_history'])

            input_node_count = SympleState.from_sympy(input_expr).node_count()
            output_node_count = output_expr.node_count()
            computed_ncr = input_node_count - output_node_count
            history_ncr = history['node_count_reduction']
            
            assert computed_ncr == history_ncr, f"Warning: Computed NCR ({computed_ncr}) doesn't match history NCR ({history_ncr})"

        returns.append(avg_return)
        avg_eval_time = batch_time / batch_size  # Calculate average evaluation time per expression
        eval_times.append(avg_eval_time)
        avg_ncr = sum([history['node_count_reduction'] for history in batch_history]) / len(batch_history)
        avg_n_steps = sum([history['n_steps'] for history in batch_history]) / len(batch_history)
        
        print(f"Epoch {epoch}/{num_epochs}, Batch {batch_number_in_epoch}/{n_batches}, Return: {avg_return:.4f}, NCR: {avg_ncr:.4f}, Batch Time: {batch_time:.4f} seconds, Avg Eval Time: {avg_eval_time:.4f} seconds, Avg Steps: {avg_n_steps:.4f}")
        # Save batch data
        batch_data = {
            'epoch': epoch,
            'batch_number_in_epoch': batch_number_in_epoch,
            'overall_batch_number': overall_batch_num,
            'avg_return': avg_return,
            'avg_ncr': avg_ncr,
            'batch_time': batch_time,
            'avg_eval_time': avg_eval_time,
            'history': batch_history
        }
        epoch_data.append(batch_data)

    # Save data after each epoch
    if epoch > 1:
        with open(json_filename, 'rb+') as f:
            f.seek(-2, os.SEEK_END)
            f.truncate()
        with open(json_filename, 'a') as f:
            f.write(',')
            json_string = json.dumps(epoch_data, indent=2)
            f.write(json_string[1:])
    else:
        with open(json_filename, 'w') as f:
            json.dump(epoch_data, f, indent=2)
    
    # Save the model state dict
    torch.save(agent.state_dict(), model_path)
    print(f"Model state dict saved to: {model_path}")

avg_time_per_batch = total_time / (num_epochs * (len(df) // batch_size))
print(f"Training completed. Average time per batch: {avg_time_per_batch:.4f} seconds")
print(f"Training data saved to: {json_filename}")
