import json
import os
import time
from typing import Dict
import torch
import pandas as pd
from sympy import sympify

from src.model.model import SympleAgent
from src.model.state import SympleState
from src.model.environment import Symple
from definitions import ROOT_DIR

# Load the dataset
dataset_path = os.path.join(ROOT_DIR, "data","dataset.json")
with open(dataset_path, "r") as f:
    df = pd.read_json(f)
df[df.columns[:3]] = df[df.columns[:3]].map(sympify)

# Initialize the agent
agent = SympleAgent(
    hidden_size=128,
    global_hidden_size=256,
    ffn_n_layers=3,
    lstm_n_layers=3
)

# Load the model
model_save_dir = os.path.join(ROOT_DIR, 'train', 'models')
model_files = [f for f in os.listdir(model_save_dir) if f.startswith('model_') and f.endswith('.pth')]

if model_files:
    # Sort model files by modification time, oldest first
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_save_dir, x)))
    
    print("Available models:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    while True:
        choice = input("Enter the number of the model you want to use (or 'latest' for the most recent): ")
        if choice.lower() == 'latest':
            model_filename = model_files[-1]
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(model_files):
            model_filename = model_files[int(choice)-1]
            break
        else:
            print("Invalid choice. Please try again.")
    
else:
    raise ValueError("No model found. Please run train.py first.")

model_path = os.path.join(model_save_dir, model_filename)
date_time_str = model_filename.split('_',1)[1].split('.')[0]
agent.load_state_dict(torch.load(model_path, weights_only=True))
print(f'Loaded model from: {os.path.basename(model_path)}')

n_attempts = 5  # Number of attempts for each example

def run_agent(expr: SympleState, agent: SympleAgent, env: Symple, **agent_forward_kwargs) -> Dict:
    agent.eval()
    start_time = time.time()
    history, final_state = agent(expr, env, **agent_forward_kwargs)
    end_time = time.time()
    
    eval_time = end_time - start_time
    ncr = sum([step['node_count_reduction'] for step in history])
    action_sequence = [(step['action_type'], step['action']) for step in history]
    
    return {
        'eval_time': eval_time,
        'ncr': ncr,
        'n_steps': len(history),
        'action_sequence': action_sequence
    }

results = []

for idx, expr in enumerate(df['expr']):
    print(f"Processing expression {idx + 1}/{len(df)}")
    attempts = []
    
    for _ in range(n_attempts):
        env = Symple()
        state = SympleState.from_sympy(sympify(expr))
        attempt_result = run_agent(state, agent, env, min_steps = 5, max_steps = 250)
        attempts.append(attempt_result)
    
    best_attempt = max(attempts, key=lambda x: x['ncr'])

    print(f"Best attempt: {best_attempt}")
    
    results.append({
        'expr': str(expr),
        'best_attempt': best_attempt,
        'all_attempts': attempts
    })

# Save results to JSON
results_dir = os.path.join(ROOT_DIR, 'src', 'benchmark', 'results')
os.makedirs(results_dir, exist_ok=True)
json_path = os.path.join(results_dir, f'benchmark_results_{date_time_str}.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {json_path}")


