import numpy as np
from sympy import Pow


def leaf_count(expr):
    args = expr.args
    if args:
        return sum(leaf_count(a) for a in args)
    else:
        return 1


def node_count(expr):
    # args = expr.args
    if not expr.is_Atom:
        return 1 + sum(leaf_count(a) for a in expr.args)
    else:
        return 1


def get_coords(expr, depth=np.inf, height=0, exclude=[]):
    if height <= 0 and not expr.func in exclude:
        coords = [[]]
    else:
        coords = []
    if expr.is_Atom or depth == 0:
        return coords

    args = list(enumerate(expr.args))
    if "exponents" in exclude and expr.func == Pow:
        args = args[:-1]
    for i, a in args:
        coords += [
            [
                i,
            ]
            + c
            for c in get_coords(a, depth=depth - 1, height=height - 1)
        ]
    return coords


def get_subexpression(expr, coord):
    if coord:
        return get_subexpression(expr.args[coord[0]], coord[1:])
    return expr


# def get_nodes(expr):
#     def _get_nodes(e, p):
#         if not e.is_Atom:
#             return [tree_node(e, p)] + sum([_get_nodes(a, e) for a in e.args], [])
#         else:
#             return [tree_node(e, p)]
#
#     return _get_nodes(expr, None)


def map_on_subexpression(function, expr, coord, *funcargs, **funckwargs):
    """
    coord: a list representing directions to traverse the tree from the trunk to some particular subexpression.

    Might be inefficient implementation... This is because sympy expressions seemingly cannot be changed in-place.
    """
    if coord:
        i = coord.pop(0)
        return expr.func(
            *(
                expr.args[:i]
                + (
                    map_on_subexpression(
                        function, expr.args[i], coord, *funcargs, **funckwargs
                    ),
                )
                + expr.args[i + 1 :]
            )
        )
    return function(expr, *funcargs, **funckwargs)
