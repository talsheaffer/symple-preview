import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from definitions import ROOT_DIR

from src.model.environment import TIME_PENALTY, NODE_COUNT_IMPORTANCE_FACTOR, COMPUTE_PENALTY_COEFFICIENT

# Find the most recent JSON file
json_dir = os.path.join(ROOT_DIR, 'train/training_data')
json_files = [f for f in os.listdir(json_dir) if f.startswith('training_data_') and f.endswith('.json')]
sorted_jsons = sorted(json_files, key=lambda x: datetime.strptime(x, 'training_data_%Y%m%d_%H%M%S.json'))
latest_json = sorted_jsons[-1]

# Parse the latest JSON filename to get the date and time
date_time_str = latest_json.split('_', 2)[2].rsplit('.', 1)[0]
parsed_date_time = datetime.strptime(date_time_str, '%Y%m%d_%H%M%S')
formatted_date_time = parsed_date_time.strftime('%Y-%m-%d %H:%M:%S')
print(f'Using training data from: {formatted_date_time}')

# Load the JSON data
with open(os.path.join(json_dir, latest_json), 'r') as f:
    data = json.load(f)

# Extract necessary data
batch_numbers = []
returns = []
eval_times = []
if 'batch_number_in_epoch' in data[0].keys():
    for entry in data:
        entry['batch_number'] = entry['batch_number_in_epoch']

for entry in data:
    epoch = entry['epoch']
    batch_in_epoch = entry['batch_number']
    batch_numbers.append((epoch - 1) * max(entry['batch_number'] for entry in data if entry['epoch'] == epoch) + batch_in_epoch)
    returns.append(entry['avg_return'])
    eval_times.append(entry['avg_eval_time'])

# Filter out |return| > 1000 and eval_time > 20
filtered_batch_numbers = []
filtered_returns = []
filtered_eval_times = []
for bn, ret, et in zip(batch_numbers, returns, eval_times):
    if abs(ret) <= 1000 and et <= 20:
        filtered_batch_numbers.append(bn)
        filtered_returns.append(ret)
        filtered_eval_times.append(et)

# Create figures subfolder if it doesn't exist
figures_dir = os.path.join(ROOT_DIR, 'train', 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Plot the return history
plt.figure(figsize=(10, 5))
plt.plot(filtered_batch_numbers, filtered_returns)
plt.title('Average Return vs Batch Number (|Return| <= 1000)')
plt.xlabel('Batch Number')
plt.ylabel('Average Return')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.savefig(os.path.join(figures_dir, 'return_history.png'))
plt.close()

# Plot the evaluation time history
plt.figure(figsize=(10, 5))
plt.plot(filtered_batch_numbers, filtered_eval_times)
plt.title('Average Evaluation Time per Expression vs Batch Number (Eval Time <= 20s)')
plt.xlabel('Batch Number')
plt.ylabel('Average Evaluation Time (seconds)')
plt.savefig(os.path.join(figures_dir, 'eval_time_history.png'))
plt.close()

# Plot both return and evaluation time on the same graph
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Batch Number')
ax1.set_ylabel('Average Return', color='blue')
ax1.plot(filtered_batch_numbers, filtered_returns, color='blue', label='Return')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Average Evaluation Time (seconds)', color='red')
ax2.plot(filtered_batch_numbers, filtered_eval_times, color='red', label='Avg Eval Time')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Average Return and Evaluation Time vs Batch Number (|Return| <= 1000, Eval Time <= 20s)')
fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'return_and_eval_time_history.png'))
plt.close()

# New plot: Moving average of return
window_size = 10
moving_avg = [sum(filtered_returns[max(0, i-window_size):i])/min(i, window_size) for i in range(1, len(filtered_returns)+1)]

plt.figure(figsize=(10, 5))
plt.plot(filtered_batch_numbers, filtered_returns, alpha=0.3, label='Raw Return')
plt.plot(filtered_batch_numbers, moving_avg, label=f'{window_size}-batch Moving Average')
plt.title(f'Raw Return and {window_size}-batch Moving Average vs Batch Number (|Return| <= 1000)')
plt.xlabel('Batch Number')
plt.ylabel('Return')
plt.legend()
plt.ylim(min(moving_avg), max(moving_avg))  # Set y-axis range based on moving average
plt.savefig(os.path.join(figures_dir, 'return_moving_average.png'))
plt.close()

# Calculate true mean node count reduction
true_ncr = [example['node_count_reduction'] for batch in data for example in batch['history']]

# Remove outliers
ncr_threshold = 150
true_ncr_filtered = [ncr for ncr in true_ncr if abs(ncr) <= ncr_threshold]
print(f"Percent of recorded NCRs removed: {(len(true_ncr) - len(true_ncr_filtered)) / len(true_ncr) * 100:.2f}%")

mean_ncr = np.mean(true_ncr_filtered)
std_ncr = np.std(true_ncr_filtered)

# Plot histogram of true node count reduction
plt.figure(figsize=(10, 5))
plt.hist(true_ncr_filtered, bins=50, edgecolor='black')
plt.title(f'Distribution of True Node Count Reduction (|NCR| <= {ncr_threshold})')
plt.xlabel('Node Count Reduction')
plt.ylabel('Frequency')
plt.axvline(mean_ncr, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ncr:.2f}\nStd: {std_ncr:.2f}')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'true_ncr_distribution.png'))
plt.close()

print(f"Mean Node Count Reduction: {mean_ncr:.2f}")
print(f"Standard Deviation of Node Count Reduction: {std_ncr:.2f}")

# Plot average node count reduction over time (batch number)
avg_ncr_per_batch = [batch['avg_ncr'] for batch in data]

# Remove outliers
avg_ncr_per_batch_filtered = [ncr if abs(ncr) <= 1000 else np.nan for ncr in avg_ncr_per_batch]

# Calculate moving average
window_size = 10
moving_avg_ncr = [np.nanmean(avg_ncr_per_batch_filtered[max(0, i-window_size):i]) for i in range(1, len(avg_ncr_per_batch_filtered)+1)]

plt.figure(figsize=(10, 5))
plt.plot(batch_numbers, avg_ncr_per_batch_filtered, alpha=0.3, label='Avg Node Count Reduction')
plt.plot(batch_numbers, moving_avg_ncr, label=f'{window_size}-batch Moving Average')
plt.title('Average Node Count Reduction vs Batch Number (|NCR| <= 1000)')
plt.xlabel('Batch Number')
plt.ylabel('Average Node Count Reduction')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.legend()
plt.ylim(min(moving_avg_ncr), max(moving_avg_ncr))  # Set y-axis range based on moving average
plt.savefig(os.path.join(figures_dir, 'avg_ncr_per_batch.png'))
plt.close()

# Extract rewards, node count reductions, and compute penalties
rewards = [time_step['reward'] for batch in data for example in batch['history'] for time_step in example['example_history']]
ncrs = [time_step['node_count_reduction'] for batch in data for example in batch['history'] for time_step in example['example_history']]
compute_penalties = [time_step['complexity'] for batch in data for example in batch['history'] for time_step in example['example_history']]
actions = [ (time_step['action_type'], time_step['action']) for batch in data for example in batch['history'] for time_step in example['example_history']]

# Extract time penalty from the first batch (assuming it's constant across all batches)
time_penalty = -0.02

# Calculate expected rewards
expected_rewards = [time_penalty - 1e-8 * cp + ncr for cp, ncr in zip(compute_penalties, ncrs)]

# Compare expected rewards with actual rewards
differences = [abs(r - er) for r, er in zip(rewards, expected_rewards)]
max_difference = max(differences)
avg_difference = sum(differences) / len(differences)

print(f"Maximum difference between expected and actual rewards: {max_difference:.6f}")
print(f"Average difference between expected and actual rewards: {avg_difference:.6f}")

# Create a dictionary to store differences for each action type
action_differences = {}

# Iterate through actions, actual rewards, and expected rewards
for action, reward, expected_reward in zip(actions, rewards, expected_rewards):
    difference = abs(reward - expected_reward)
    if difference !=0:
    
        # If the action is not in the dictionary, add it
        if action not in action_differences:
            action_differences[action] = []
        
        # Append the difference for this action
        action_differences[action].append(difference)

# Calculate average difference for each action
avg_differences = {action: sum(diffs) / len(diffs) for action, diffs in action_differences.items()}

num_discrepancies = {action: len(diffs) for action, diffs in action_differences.items()}

print(f"Number of discrepancies for each action: {num_discrepancies}")

# Sort actions by average difference in descending order
sorted_actions = sorted(avg_differences.items(), key=lambda x: x[1], reverse=True)

# Print the results
print("\nAverage difference between expected and actual rewards for each action:")
for action, avg_diff in sorted_actions:
    print(f"Action: {action}, Average Difference: {avg_diff:.6f}")

# Plot the top 10 actions with the highest average differences
top_10_actions = sorted_actions[:10]
action_names = [f"{action[0]}: {action[1]}" for action, _ in top_10_actions]
avg_diffs = [avg_diff for _, avg_diff in top_10_actions]

# Filter external actions
external_actions = [action[1] for action in actions if action[0] not in ['high_level', 'internal']]

# Create histogram
plt.figure(figsize=(12, 6))
plt.hist(external_actions, bins=len(set(external_actions)), edgecolor='black')
plt.title('Histogram of External Actions')
plt.xlabel('Action')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot
plt.savefig(os.path.join(figures_dir, 'external_actions_histogram.png'))
plt.close()

print("Histogram of external actions has been saved as 'external_actions_histogram.png'")
