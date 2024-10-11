import time
import os
import yaml

import sympy as sp

import pandas as pd
import json
from datetime import datetime

import torch

# from src.model.environment import Symple
from src.model.model import SympleAgent, NUM_INTERNAL_OPS
from src.model.training import train_on_batch
from src.model.state import SympleState
from src.model.environment import TIME_PENALTY, COMPUTE_PENALTY_COEFFICIENT

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
metadata_filename = os.path.join(training_data_dir, f"metadata_{timestamp}.yaml")

model_save_dir = os.path.join(ROOT_DIR, 'train', 'models')
os.makedirs(model_save_dir, exist_ok=True)
model_filename = f'model_{timestamp}'
model_save_path = os.path.join(model_save_dir, model_filename)

model_files = [f for f in os.listdir(model_save_dir) if f.startswith('model_') and f.endswith('.pth')]

if model_files:
    # Sort model files by modification time, oldest first
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_save_dir, x)))
    
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    while True:
        choice = input("Enter the number of the model you want to use (or 'none' for untrained model or 'latest' for the most recent): ")
        if choice.lower() == 'none':
            model_path = None
            break
        elif choice.lower() == 'latest':
            model_path = os.path.join(model_save_dir, model_files[-1])
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(model_files):
            model_path = os.path.join(model_save_dir, model_files[int(choice)-1])
            break
        else:
            print("Invalid choice. Please try again.")
else:
    model_path = None
    print('No previous model found.')

agent = SympleAgent(
    hidden_size = 128,
    global_hidden_size=256,
    ffn_n_layers=3,
    lstm_n_layers=3,
    temperature=0.2
)

if model_path:
    agent.load_state_dict(torch.load(model_path, weights_only=True))
    print(f'Loaded model from: {os.path.basename(model_path)}')
else:
    print('Using untrained model.')

def save_model(model,suffix=''):
    torch.save(model.state_dict(), model_save_path+suffix+'.pth')
    print(f"Model state dict saved to: {model_save_path+suffix+'.pth'}")

# Define learning rate schedule
initial_lr = 0.001
lr_decay_factor = 1.0

# Initialize Adam optimizer
weight_decay = 0.001
optimizer = torch.optim.Adam(
    agent.parameters(),
    lr=initial_lr,
    weight_decay = weight_decay
)
# Training loop
num_epochs = 30
batch_size = 32

metadata = {
    'model_hyperparameters': {
        'hidden_size': agent.hidden_size,
        'global_hidden_size': agent.global_hidden_size,
        'ffn_n_layers': agent.ffn.n_layers,
        'lstm_n_layers': agent.lstm.num_layers,
        'num_internal_ops': NUM_INTERNAL_OPS,
        'num_external_ops': agent.num_ops,
        'temperature': agent.temperature
    },
    'last_training_epoch': 0,
    'environment_parameters': {
        'time_penalty': TIME_PENALTY,
        'compute_penalty_coefficient': COMPUTE_PENALTY_COEFFICIENT
    },
    'training_parameters': {
        'behavior_policy': str(None),
        'min_steps': 5,
        'max_steps': 250,
        'initial_lr': initial_lr,
        'lr_decay_factor': lr_decay_factor,
        'batch_size': batch_size,
        'num_epochs': num_epochs
    },
    'optimizer': {
        'type': 'Adam',
        'lr': initial_lr,
        'current_lr': initial_lr,
        'weight_decay': weight_decay
    }
}

with open(metadata_filename, 'w') as f:
    yaml.dump(metadata, f)

# Save model parameters in a metadata file
model_params_filename = os.path.join(model_save_dir, f"model_hyperparams_{timestamp}.yaml")
model_params = metadata['model_hyperparameters']

with open(model_params_filename, 'w') as f:
    yaml.dump(model_params, f)

print(f"Model parameters saved to: {model_params_filename}")


total_time = 0
returns = []
total_time = 0
eval_times = []

training_data = []

overall_batch_num = 0

for epoch in range(1, num_epochs + 1):
    # behavior_policy = ('temperature', 1.5 + .2 * (epoch - 1)) if epoch < 10 else None
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
        avg_return, batch_history, output_expr_nodes, _ = train_on_batch(
            agent,
            batch,
            optimizer,
            behavior_policy=behavior_policy,
            agent_forward_kwargs=dict(
                min_steps=5,
                max_steps=250,
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
        
        if avg_ncr > 26:
            save_model(agent,suffix=f'_{epoch}_{batch_number_in_epoch}')

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
    save_model(agent)

    metadata['last_training_epoch'] = epoch
    metadata['training_parameters']['behavior_policy'] = str(behavior_policy)
    metadata['optimizer']['current_lr'] = current_lr
    
    with open(metadata_filename, 'w') as f:
        yaml.dump(metadata, f)

avg_time_per_batch = total_time / (num_epochs * (len(df) // batch_size))
print(f"Training completed. Average time per batch: {avg_time_per_batch:.4f} seconds")
print(f"Training data saved to: {json_filename}")
print(f"Metadata saved to: {metadata_filename}")
