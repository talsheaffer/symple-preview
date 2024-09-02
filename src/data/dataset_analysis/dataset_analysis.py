import pandas as pd
from sympy import sympify
from matplotlib import pyplot as plt

from definitions import ROOT_DIR
with open(ROOT_DIR+'/data/dataset.json','r') as f:
    df = pd.read_json(f)
df[df.columns[:3]] = df[df.columns[:3]].map(sympify)

# df.head()

# [display(*d) for d in df[['expr','simp']].head().values]

# @title difficulty


df['difficulty'].plot(kind='hist', bins=20, title='difficulty')
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.savefig("difficulty.png")

# @title node count expr vs node count simp

df.plot(kind='scatter', x='node count expr', y='node count simp', s=6, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.gca().set_aspect('equal')
plt.savefig("node_count_comp_vs_simp.png")

# @title node count simp vs node count simplified

df.plot(kind='scatter', x='node count simplified', y='node count simp', s=6, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)
plt.gca().set_aspect('equal')
plt.savefig("node_count_simp_vs_simplified.png")