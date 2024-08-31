import os

from itertools import product

from random import randint

from src.data.utils.complicate import complicate
from src.utils.tree_iter import node_count

from sympy import symbols, simplify

x, y, z = symbols('x, y, z')

import pandas as pd

configs = [
    {
        "rand gen args": (3, vars),
        "number of complications": n,
        "number of datapoints": 100
    } for vars, n in product([
        (x,),
        (x, y),
        (x, y, z)
    ], [
        5,
        7
    ])
]

datapoints = []
for config in configs:
    for _ in range(config["number of datapoints"]):
        esimp, ecomp = complicate(
            n=config["number of complications"],
            rand_gen_args=(lambda rga: (rga[0], (randint(1, 5),) + rga[1]))(config["rand gen args"])
        )
        secomp = simplify(ecomp)
        datapoints.append({
            "expr": ecomp,
            "simp": esimp,
            "simplified": secomp,
            "node count expr": node_count(ecomp),
            "node count simp": node_count(esimp),
            "node count simplified": node_count(secomp)
        })

ds = pd.DataFrame.from_dict(datapoints)
ds[ds.columns[:3]] = ds[ds.columns[:3]].map(str)
ds['difficulty'] = ds['node count expr'] - ds['node count simp']

from definitions import ROOT_DIR
file_path = ROOT_DIR+"/data/dataset.json"
with open(file_path, 'w') as f:
    ds.to_json(f)

# with open('data.json','r') as f:
#     ds = pd.read_json(f)
