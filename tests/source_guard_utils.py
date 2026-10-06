"""Structural assertions about source files, for tests that check wiring.

Several tests in this suite verify that a call is guarded, or that one branch
returns before another, by regex-matching the source with a character budget
(`if local_rank == 0:[\\s\\S]{0,200}write_...`). A budget like that fails the
moment any unrelated line is inserted between the guard and the call, which
reports a regression that did not happen and pressures the next person to widen
the number instead of checking the property.

These helpers assert the same intent by parsing the source, so an insertion
between the guard and the call is not an error while an unguarded call still is.
"""
from __future__ import annotations

import ast


def _is_rank0_test(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "local_rank"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 0
    )


def calls_under_rank0_guard(src: str, call_name: str) -> bool:
    """True if some call to `call_name` is inside an `if local_rank == 0:` body."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_rank0_test(node.test):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == call_name):
                return True
    return False


def target_only_branch_returns_before_spec_loop(src: str) -> bool:
    """True if the `cfg.spec_steps == 0` branch returns before the spec loop.

    The property under test is ordering: the target-only baseline must
    early-return rather than fall through into speculative decoding. Comparing
    line positions asserts that directly, where a bounded-window regex only
    asserted that the two lines were close together.
    """
    lines = src.splitlines()
    branch = loop = None
    for i, line in enumerate(lines):
        if branch is None and line.strip().startswith("if cfg.spec_steps == 0:"):
            branch = i
        if loop is None and "Main speculative decoding loop" in line:
            loop = i
    if branch is None or loop is None:
        return False
    for line in lines[branch:loop]:
        if line.strip().startswith("return generated_ids, metrics"):
            return True
    return False
