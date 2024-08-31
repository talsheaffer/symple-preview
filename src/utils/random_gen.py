import numpy as np
from random import choice, randint
from sympy import FunctionClass, Add, Mul, cos, sin, binomial, arity, S
from src.utils.tree_iter import get_coords, get_subexpression


def random_args(n, atoms, funcs, evaluate=True):
    a = funcs + atoms
    g = []
    for _ in range(n):
        ai = choice(a)
        if isinstance(ai, FunctionClass):
            g.append(ai(*random_args(arity(ai), atoms, funcs), evaluate=evaluate))
        else:
            g.append(ai)
    return g


def random_expr(ops, atoms, funcs=(), evaluate=True, inversion_prob=.1):
    types = [Add, Mul]
    atoms = tuple(atoms)
    while 1:
        e = (S.Zero)
        while e.count_ops() < ops:
            _ = choice(types)(
                *random_args(randint(1, 3), atoms, funcs, evaluate=evaluate),
                evaluate=evaluate
            )
            e = choice(types)(e, _ ** np.random.choice(
                (1, -1),
                p=(1 - inversion_prob, inversion_prob)
            ),
                              evaluate=evaluate
                              )
            if e is S.NaN: break
        else:
            return e


def random_expr_if_none(expr, *args, **kwargs):
    return expr if expr is not None else random_expr(*args, **kwargs)


def expr_grabber(l, *args, **kwargs):
    """
    l should have the form [None, (actual expressions)]
    """
    prob = .1

    def random_from_list():
        p = np.ones(len(l)) * (1 - prob) / len(l)
        p[0] += prob
        expr = np.random.choice(l, p=p)
        return expr if expr is not None else random_expr(*args, **kwargs)

    return random_from_list


def random_coord(expr, **kwargs):
    return choice(
        get_coords(expr, **kwargs)
    )


def random_subexpression(expr, **kwargs):
    coord = random_coord(expr, **kwargs)
    return (coord, get_subexpression(expr, coord))
