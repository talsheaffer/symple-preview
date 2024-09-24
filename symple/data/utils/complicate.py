import inspect
from random import randint

import numpy as np
from sympy import expand, symbols

from symple.utils.random_gen import expr_grabber

x, y, z = symbols("x, y, z")


def gen_trick_1(e1, e2):
    return (e1 * e2, expand(e1 * e2))


def gen_trick_2(e1, e2):
    return (1 / e1 + 1 / e2, (e1 + e2) / (expand(e1 * e2)))


gen_tricks = [gen_trick_1, gen_trick_2]


def generate_simplifiable(eg):
    trick = np.random.choice(gen_tricks)
    return trick(*(eg() for _ in inspect.signature(trick).parameters))


def comp_trick_1(e, eg):
    ep = eg()
    return expand(e * ep) / ep


def comp_trick_2(e, eg):
    ep = eg()
    return e * (1 + ep) - expand(e * ep)


def comp_trick_3(e, eg):
    ep = eg()
    return expand(e * ep) / ep


comp_tricks = [comp_trick_1, comp_trick_2, comp_trick_3]


# First approach - uses "comp tricks" to obtain equivalent expression. Currently seems to be to easy for "simplify"
# def complicate(n=3):
#     l = [None]
#     eg = expr_grabber(l,3, (randint(1,5),x,y))
#     esimp, ecomp = generate_simplifiable(eg)
#     print(f"The node count of the simple expression is: {node_count(esimp)}")
#     print(f"The node count of the complicated expression is: {node_count(ecomp)}")
#     l += list(esimp.args)
#     for _ in range(n):
#         trick = np.random.choice(comp_tricks)
#         coord, subexp = random_subexpression(
#             ecomp, depth=3, height =1, exclude = ["exponents"]
#         )
#         ecomp, l = (
#             map_on_subexpression(
#                 lambda x: trick(x, eg),
#                 ecomp,
#                 # random_node(ecomp,depth=3, height =1),
#                 coord
#             ),
#             # trick(ecomp, eg),
#             l+[subexp]
#         )
#         # l = [None]+l[-6:]
#         print(f"The node count of the complicated expression is: {node_count(ecomp)}")
#     return esimp, ecomp

# Second approach - only generates new, inequivalent expressions. Seems to be harder to crack


def complicate(n=7, rand_gen_args=(3, (randint(1, 5), x, y)), **kwargs):
    list_of_expressions = [None]
    eg = expr_grabber(list_of_expressions, *rand_gen_args, **kwargs)
    for _ in range(n):
        esimp, ecomp = generate_simplifiable(eg)
        # print(f"The node count of the simple expression is: {node_count(esimp)}")
        # print(f"The node count of the complicated expression is: {node_count(ecomp)}")
        list_of_expressions += list(esimp.args) + [esimp]
        # make sure not to use too simple subexpressions:
        list_of_expressions = [None] + list_of_expressions[-6:]
    return esimp, ecomp
