"""Contains an AST-based math expression resolver for omegaconf/hydra.

Ever wondered if it is possible to solve simple math equations in an OmegaConf config (or in a
Hydra config)? well, this is the solution: simply import this module in your python app where you
intend to use omegaconf (or Hydra), and then use the `{ast_eval:'EXPRESSION'}` resolver.

Enjoy!
"""
import ast
import operator as op

import omegaconf

supported_ast_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
}

supported_ast_nodes = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    *supported_ast_operators.keys(),
)


def _ast_eval_expr(node):
    """Evaluates part of an expression (potentially recursively)."""
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return supported_ast_operators[type(node.op)](  # noqa
            _ast_eval_expr(node.left),
            _ast_eval_expr(node.right),
        )
    elif isinstance(node, ast.UnaryOp):
        return supported_ast_operators[type(node.op)](  # noqa
            _ast_eval_expr(node.operand)
        )
    else:
        raise ValueError(f"unsupported operation: {type(node)}")


def ast_eval(expression: str):
    """Evaluates a simple arithmetic expression using the AST package."""
    node = ast.parse(expression, mode="eval")
    subnodes = list(ast.walk(node))
    node_types_supported = [isinstance(n, supported_ast_nodes) for n in subnodes]
    if not all(node_types_supported):
        unsupported_types = ', '.join([
            str(type(subnodes[nidx]))
            for nidx, supported in enumerate(node_types_supported)
            if not supported
        ])
        raise ValueError(
            "invalid expression; only simple arithmetic ops are supported\n"
            f"found invalid expression node type(s): {unsupported_types}"
        )
    return _ast_eval_expr(node.body)


omegaconf.OmegaConf.register_new_resolver("ast_eval", ast_eval)


# Example usage:

assert ast_eval("2 * (-4 + 15 / 2)") == 7

config = omegaconf.OmegaConf.create({
    "a": 4,
    "b": 15,
    "result": "${ast_eval:'2 * (-${a} + ${b} / 2)'}",
})
assert config.result == 7

