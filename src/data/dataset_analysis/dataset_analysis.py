import pandas as pd
from sympy import sympify
from matplotlib import pyplot as plt
import os

from src.model.state import SympleState

from definitions import ROOT_DIR

with open(ROOT_DIR + "/data/dataset.json", "r") as f:
    df = pd.read_json(f)
df[df.columns[:3]] = df[df.columns[:3]].map(sympify)

# df.head()

# [display(*d) for d in df[['expr','simp']].head().values]

# @title difficulty

output_dir = os.path.join(ROOT_DIR, "src/data/dataset_analysis")
os.makedirs(output_dir, exist_ok=True)

difficulty_mean = df["difficulty"].mean()
difficulty_std = df["difficulty"].std()

plt.figure(figsize=(10, 6))
df["difficulty"].hist(bins=20)
plt.axvline(difficulty_mean, color='r', linestyle='dashed', linewidth=2)
plt.title(f"Difficulty Distribution\nMean: {difficulty_mean:.2f}, Std: {difficulty_std:.2f}")
plt.xlabel("Difficulty")
plt.ylabel("Frequency")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.savefig(os.path.join(output_dir, "difficulty.png"))
plt.close()

# @title node count expr vs node count simp

df.plot(kind="scatter", x="node count expr", y="node count simp", s=6, alpha=0.8)
plt.gca().spines[
    [
        "top",
        "right",
    ]
].set_visible(False)
plt.gca().set_aspect("equal")
plt.savefig(os.path.join(output_dir, "node_count_comp_vs_simp.png"))
plt.close()

# @title node count simp vs node count simplified

df.plot(kind="scatter", x="node count simplified", y="node count simp", s=6, alpha=0.8)
plt.gca().spines[
    [
        "top",
        "right",
    ]
].set_visible(False)
plt.gca().set_aspect("equal")
plt.savefig(os.path.join(output_dir, "node_count_simp_vs_simplified.png"))
plt.close()

# @title node count expr vs node count simplified

df.plot(kind="scatter", x="node count expr", y="node count simplified", s=6, alpha=0.8)
plt.gca().spines[
    [
        "top",
        "right",
    ]
].set_visible(False)
plt.gca().set_aspect("equal")
plt.savefig(os.path.join(output_dir, "node_count_expr_vs_simplified.png"))
plt.close()


# Compute average and std for node count difference, simplified vs expr
node_count_diff = df['node count expr'] - df['node count simplified']
avg_node_count_diff = node_count_diff.mean()
std_node_count_diff = node_count_diff.std()

print(f"Average node count difference (expr - simplified): {avg_node_count_diff:.2f}")
print(f"Standard deviation of node count difference: {std_node_count_diff:.2f}")

# Visualize the distribution of node count difference
plt.figure(figsize=(10, 6))
plt.hist(node_count_diff, bins=30, edgecolor='black')
plt.axvline(avg_node_count_diff, color='r', linestyle='dashed', linewidth=2)
plt.title(f'Distribution of Node Count Difference (expr - simplified)\nMean: {avg_node_count_diff:.2f}, Std: {std_node_count_diff:.2f}')
plt.xlabel('Node Count Difference')
plt.ylabel('Frequency')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig(os.path.join(output_dir, "node_count_diff_distribution.png"))
plt.close()


# Compute node count for ExprNodes

def get_expr_node_count(expr):
    return SympleState.from_sympy(expr).node_count()

df['expr_node_count'] = df['expr'].apply(get_expr_node_count)
df['simplified_expr_node_count'] = df['simplified'].apply(get_expr_node_count)

# Plot node count vs node count simplified for ExprNodes
plt.figure(figsize=(10, 6))
plt.scatter(df['simplified_expr_node_count'], df['expr_node_count'], s=6, alpha=0.8)
plt.title('ExprNode Count vs ExprNode Count Simplified')
plt.xlabel('ExprNode Count Simplified')
plt.ylabel('ExprNode Count')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.gca().set_aspect('equal')
plt.savefig(os.path.join(output_dir, "expr_node_count_vs_simplified.png"))
plt.close()

# Compute average and std for ExprNode count difference
expr_node_count_diff = df['expr_node_count'] - df['simplified_expr_node_count']
avg_expr_node_count_diff = expr_node_count_diff.mean()
std_expr_node_count_diff = expr_node_count_diff.std()

print(f"Average ExprNode count difference (expr - simplified): {avg_expr_node_count_diff:.2f}")
print(f"Standard deviation of ExprNode count difference: {std_expr_node_count_diff:.2f}")

# Visualize the distribution of ExprNode count difference
plt.figure(figsize=(10, 6))
plt.hist(expr_node_count_diff, bins=30, edgecolor='black')
plt.axvline(avg_expr_node_count_diff, color='r', linestyle='dashed', linewidth=2)
plt.title(f'Distribution of ExprNode Count Difference (expr - simplified)\nMean: {avg_expr_node_count_diff:.2f}, Std: {std_expr_node_count_diff:.2f}')
plt.xlabel('ExprNode Count Difference')
plt.ylabel('Frequency')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig(os.path.join(output_dir, "expr_node_count_diff_distribution.png"))
plt.close()
