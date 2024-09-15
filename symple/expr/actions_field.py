from symple.expr.expr_node import ARG_NULL, ExprNode, ExprNodeType, minus_one, one, zero


def can_commute(expr: ExprNode) -> bool:
    # Field Axiom
    return expr.type in (ExprNodeType.ADD_TYPE, ExprNodeType.MUL_TYPE)

def can_associate_b(expr: ExprNode) -> bool:
    # Field Axiom
    return (expr.type == ExprNodeType.ADD_TYPE and expr.b.type == ExprNodeType.ADD_TYPE) or (
        expr.type == ExprNodeType.MUL_TYPE and expr.b.type == ExprNodeType.MUL_TYPE
    )

def can_distribute_b(expr: ExprNode) -> bool:
    # Field Axiom
    return (expr.type == ExprNodeType.MUL_TYPE and expr.b.type == ExprNodeType.ADD_TYPE) or (
        expr.type == ExprNodeType.POW_TYPE and expr.b.type == ExprNodeType.MUL_TYPE
    )

def can_undistribute_b(expr: ExprNode) -> bool:
    # Field Axiom
    return (
        expr.type == ExprNodeType.ADD_TYPE
        and expr.a.type == ExprNodeType.MUL_TYPE
        and expr.b.type == ExprNodeType.MUL_TYPE
        and expr.a.a == expr.b.a
    ) or (
        expr.type == ExprNodeType.MUL_TYPE
        and expr.a.type == ExprNodeType.POW_TYPE
        and expr.b.type == ExprNodeType.POW_TYPE
        and expr.a.a == expr.b.a
    )

def can_reduce_unit(expr: ExprNode) -> bool:
    # Field Axiom
    return (
        (
            expr.type == ExprNodeType.ADD_TYPE 
            and (expr.a == zero or expr.b == zero)
        ) or (
            expr.type == ExprNodeType.MUL_TYPE 
            and (expr.a == one or expr.b == one)
        ) or (
            expr.type == ExprNodeType.POW_TYPE
            and (expr.b == zero or expr.b == one or expr.a == zero or expr.a == one)
        )
    )
    
def can_multiply_unit(expr: ExprNode) -> bool:
    # Field Property
    return True

def can_add_unit(expr: ExprNode) -> bool:
    # Field Property
    return True

def can_cancel(expr: ExprNode) -> bool:
    # Field Property
    if expr.type != ExprNodeType.ADD_TYPE:
        return False

    def can_cancel_b(a, b):
        # expect b = (-1) * b.b or b = b.b * (-1)
        if b.type != ExprNodeType.MUL_TYPE:
            return False
        ba, bb = b.a, b.b
        if b.a != minus_one:
            ba, bb = bb, ba
        return a == bb

    return can_cancel_b(expr.a, expr.b) or can_cancel_b(expr.b, expr.a)

def commute(expr: ExprNode) -> ExprNode:
    # Field Axiom
    if not can_commute(expr):
        raise ValueError(f"Cannot commute {expr.type}.")

    return ExprNode(
        type=expr.type,
        arg=ARG_NULL,
        p=expr.p,
        a=expr.b,
        b=expr.a,
    )

def associate_b(expr: ExprNode  ) -> ExprNode:
    # Field Axiom
    if not can_associate_b(expr):
        raise ValueError(f"Cannot associate {expr.type}.")

    a, b = expr.a, expr.b
    expr = ExprNode(
        type=expr.type,
        arg=ARG_NULL,
        p=expr.p,
        a=ExprNode(
            type=expr.type,
            arg=ARG_NULL,
            a=a,
            b=b.a,
        ),
        b=b.b,
    )
    return expr

def distribute_b(expr: ExprNode) -> ExprNode:
    # Field Axiom
    if not can_distribute_b(expr):
        raise ValueError(f"Cannot distribute {expr.type}.")

    a, b = expr.a, expr.b
    return ExprNode(
        type=b.type,
        arg=ARG_NULL,
        p=expr.p,
        a=ExprNode(
            type=expr.type,
            arg=ARG_NULL,
            a=a,
            b=b.a,
        ),
        b=ExprNode(
            type=expr.type,
            arg=ARG_NULL,
            a=a,
            b=b.b,
        ),
    )

def undistribute_b(expr: ExprNode) -> ExprNode:
    # Field Axiom
    if not can_undistribute_b(expr):
        raise ValueError(f"Cannot undistribute {expr.type}.")

    a, b = expr.a, expr.b
    return ExprNode(
        type=a.type,
        arg=ARG_NULL,
        p=expr.p,
        a=a.a,
        b=ExprNode(
            type=expr.type,
            arg=ARG_NULL,
            a=a.b,
            b=b.b,
        ),
    )

def reduce_unit(expr: ExprNode) -> ExprNode:
    # Field Axiom
    if not can_reduce_unit(expr):
        raise ValueError(f"Cannot reduce unit {expr.type}.")

    a, b = expr.a, expr.b
    if expr.type == ExprNodeType.ADD_TYPE:
        return b if a == zero else a
    elif expr.type == ExprNodeType.MUL_TYPE:
        return b if a == one else a
    elif expr.type == ExprNodeType.POW_TYPE:
        if b == zero:
            return one
        elif b == one:
            return expr.a
        elif a == zero:
            return zero
        elif a == one:
            return one

    raise ImportError("Something went wrong")

def multiply_unit(expr: ExprNode) -> ExprNode:
    # Field Property
    if not can_multiply_unit(expr):
        raise ValueError(f"Cannot multiply unit {expr.type}.")

    return ExprNode(
        type=ExprNodeType.MUL_TYPE,
        arg=ARG_NULL,
        p=expr.p,
        a=one.clone(),
        b=expr.clone(),
    )
    
def add_unit(expr: ExprNode) -> ExprNode:
    if not can_add_unit(expr):
        raise ValueError(f"Cannot add unit {expr.type}.")

    return ExprNode(
        type=ExprNodeType.ADD_TYPE,
        arg=ARG_NULL,
        p=expr.p,
        a=zero.clone(),
        b=expr.clone(),
    )

def cancel(expr: ExprNode) -> ExprNode:
    # Field Property
    if not can_cancel(expr):
        raise ValueError(f"Cannot cancel {expr.type}.")

    return zero