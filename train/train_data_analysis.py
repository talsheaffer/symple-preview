import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from definitions import ROOT_DIR

# Find the most recent JSON file
json_dir = os.path.join(ROOT_DIR, 'train/training_data')
json_files = [f for f in os.listdir(json_dir) if f.startswith('training_data_') and f.endswith('.json')]
latest_json = max(json_files, key=lambda x: datetime.strptime(x, 'training_data_%Y%m%d_%H%M%S.json'))

# Load the JSON data
with open(os.path.join(json_dir, "/Users/talsheaffer/My Drive/code/ml/sympy_simp/symple/train/training_data/training_data_20240916_180411.json"), 'r') as f:
    data = json.load(f)

# Extract necessary data
batch_numbers = []
returns = []
eval_times = []
for entry in data:
    epoch = entry['epoch']
    batch_in_epoch = entry['batch_number']
    batch_numbers.append((epoch - 1) * max(entry['batch_number'] for entry in data if entry['epoch'] == epoch) + batch_in_epoch)
    returns.append(entry['avg_return'])
    eval_times.append(entry['avg_eval_time'])

# Create figures subfolder if it doesn't exist
figures_dir = os.path.join(ROOT_DIR, 'train', 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Plot the return history
plt.figure(figsize=(10, 5))
plt.plot(batch_numbers, returns)
plt.title('Average Return vs Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Average Return')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.savefig(os.path.join(figures_dir, 'return_history.png'))
plt.close()

# Plot the evaluation time history
plt.figure(figsize=(10, 5))
plt.plot(batch_numbers, eval_times)
plt.title('Average Evaluation Time per Expression vs Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Average Evaluation Time (seconds)')
plt.savefig(os.path.join(figures_dir, 'eval_time_history.png'))
plt.close()

# Plot both return and evaluation time on the same graph
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Batch Number')
ax1.set_ylabel('Average Return', color='blue')
ax1.plot(batch_numbers, returns, color='blue', label='Return')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Average Evaluation Time (seconds)', color='red')
ax2.plot(batch_numbers, eval_times, color='red', label='Avg Eval Time')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Average Return and Evaluation Time vs Batch Number')
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'return_and_eval_time_history.png'))
plt.close()

# New plot: Moving average of return
window_size = 10
moving_avg = [sum(returns[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(returns)+1)]

plt.figure(figsize=(10, 5))
plt.plot(batch_numbers, returns, alpha=0.3, label='Raw Return')
plt.plot(batch_numbers, moving_avg, label=f'{window_size}-batch Moving Average')
plt.title(f'Raw Return and {window_size}-batch Moving Average vs Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Return')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'return_moving_average.png'))
plt.close()
