import json
import statistics
from typing import List

import sympy as sp
import z3

from src.utils.tree_iter import node_count

DATA = "data/dataset.json"

A_LOWER_ASCII = ord("a")
BOLD_START = "\033[1m"
BOLD_END = "\033[0m"


consts = {
    "x": sp.Symbol("x"),
    "y": sp.Symbol("y"),
    "z": sp.Symbol("z"),
}


def z3_to_sympy(z3_expr):
    if z3.is_const(z3_expr):
        return sp.Symbol(str(z3_expr))
    elif z3.is_app(z3_expr):
        op = z3_expr.decl().name()
        args = [z3_to_sympy(arg) for arg in z3_expr.children()]
        if op == "+":
            return sp.Add(*args, evaluate=False)
        elif op == "-":
            return sp.Add(args[0], -args[1], evaluate=False)
        elif op == "*":
            return sp.Mul(*args, evaluate=False)
        elif op == "/":
            return sp.Mul(args[0], sp.Pow(args[1], -1, evaluate=False), evaluate=False)
        elif op == "^":
            return sp.Pow(args[0], args[1], evaluate=False)
        else:
            raise NotImplementedError(f"Unsupported Z3 operator: {op} in {z3_expr}")
    else:
        raise NotImplementedError("Unsupported Z3 expression type")


def sympy_to_z3(sympy_expr):
    if isinstance(sympy_expr, sp.Symbol):
        return z3.Const(str(sympy_expr), z3.RealSort())
    elif isinstance(sympy_expr, sp.Add):
        args = [sympy_to_z3(arg) for arg in sympy_expr.args]
        return z3.Sum(args)
    elif isinstance(sympy_expr, sp.Mul):
        args = [sympy_to_z3(arg) for arg in sympy_expr.args]
        return z3.Product(args)
    elif isinstance(sympy_expr, sp.Pow):
        base = sympy_to_z3(sympy_expr.base)
        exponent = sympy_to_z3(sympy_expr.exp)
        return base**exponent
    elif isinstance(sympy_expr, sp.Integer):
        return z3.IntVal(sympy_expr)
    else:
        raise NotImplementedError(
            "Unsupported SymPy expression type: " + str(type(sympy_expr))
        )


def metric(expr: sp.Expr) -> int:
    return node_count(expr)


def loss(x: sp.Expr, y: sp.Expr) -> int:
    return metric(y) / metric(x)


def simplify_sympy(exprs: List[sp.Expr]) -> List[sp.Expr]:
    return [sp.simplify(e) for e in exprs]


def simplify_z3(exprs: List[sp.Expr]) -> List[sp.Expr]:
    simp = []
    for e_sp in exprs:
        e_z3 = sympy_to_z3(e_sp)
        e_z3_simp = z3.simplify(e_z3)
        try:
            simp.append(z3_to_sympy(e_z3_simp))
        except:
            simp.append(None)
    return simp


def simplify_z3_context(exprs: List[sp.Expr]) -> List[sp.Expr]:
    simp = []
    for e_sp in exprs:
        e_z3 = sympy_to_z3(e_sp)
        e_z3_simp = z3.simplify(e_z3, local_ctx=True)
        try:
            simp.append(z3_to_sympy(e_z3_simp))
        except:
            simp.append(None)
    return simp


def run_benchmark():
    data = json.load(open(DATA, "r"))
    X = [sp.sympify(e, evaluate=False) for e in data["expr"].values()]

    models = {
        "sympy": simplify_sympy,
        "z3": simplify_z3,
        "z3_context": simplify_z3_context,
    }
    print(f"Expressions: {len(X)}")
    print()
    for name, model in models.items():
        Y = model(X)
        ratios = [loss(x, y) for x, y in zip(X, Y) if y is not None]
        mean_ratio = statistics.mean(ratios)
        std_ratio = statistics.stdev(ratios)

        print(BOLD_START + name + BOLD_END)
        print("Failed: ", len(X) - len(ratios))
        print(f"Mean: {100 * mean_ratio:.2f}%")
        print(f"Std:  {100 * std_ratio:.2f}%")
        print()


run_benchmark()
