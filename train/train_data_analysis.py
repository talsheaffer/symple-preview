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
latest_json = max(json_files, key=lambda x: datetime.strptime(x, 'training_data_%Y%m%d_%H%M%S.json'))

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



# Calculate true mean node count reduction
true_ncr = [batch['node_count_reductions'] for batch in data]
true_ncr = [item for sublist in true_ncr for item in sublist]  # Flatten the list

mean_ncr = np.mean(true_ncr)
std_ncr = np.std(true_ncr)

# Plot histogram of true node count reduction
plt.figure(figsize=(10, 5))
plt.hist(true_ncr, bins=50, edgecolor='black')
plt.title('Distribution of True Node Count Reduction')
plt.xlabel('Node Count Reduction')
plt.ylabel('Frequency')
plt.axvline(mean_ncr, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_ncr:.2f}\nStd: {std_ncr:.2f}')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'true_ncr_distribution.png'))
plt.close()

print(f"Mean Node Count Reduction: {mean_ncr:.2f}")
print(f"Standard Deviation of Node Count Reduction: {std_ncr:.2f}")



# Extract rewards, node count reductions, and compute penalties
rewards = [time_step['reward'] for batch in data for hist in batch['history'] for time_step in hist]
ncrs = [time_step['node_count_reduction'] for batch in data for hist in batch['history'] for time_step in hist]
compute_penalties = [time_step['complexity'] for batch in data for hist in batch['history'] for time_step in hist]
actions = [ (time_step['action_type'], time_step['action']) for batch in data for hist in batch['history'] for time_step in hist]

# Extract time penalty from the first batch (assuming it's constant across all batches)
time_penalty = TIME_PENALTY

# Calculate expected rewards
expected_rewards = [time_penalty - COMPUTE_PENALTY_COEFFICIENT * cp + NODE_COUNT_IMPORTANCE_FACTOR * ncr for cp, ncr in zip(compute_penalties, ncrs)]

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
