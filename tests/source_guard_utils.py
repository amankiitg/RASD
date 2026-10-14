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


def _iter_enclosing(src: str, needle: str, guard_fragment: str):
    """Yield (line_index, guard_indent) for guards enclosing `needle`."""
    lines = src.splitlines()
    target = next((i for i, ln in enumerate(lines) if needle in ln), None)
    if target is None:
        return
    target_indent = len(lines[target]) - len(lines[target].lstrip())
    for i in range(target - 1, -1, -1):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent >= target_indent:
            # A statement at or inside the target's own indent is not a guard.
            continue
        if stripped.startswith("if ") and guard_fragment in stripped:
            yield i, indent
            return


def call_gated_by(src: str, needle: str, guard_fragment: str) -> bool:
    """True if the first line containing `needle` sits inside an enclosing guard.

    Replaces "the guard must be within the previous N lines", which fails on any
    unrelated insertion and reports a regression that did not happen. Here an
    insertion between the guard and the call is not an error and an unguarded
    call still is.
    """
    return next(_iter_enclosing(src, needle, guard_fragment), None) is not None


def assignment_precedes_break(src: str, assign_needle: str,
                              guard_fragment: str | None = None) -> bool:
    """True if `assign_needle` is assigned before a `break` in the same block.

    Used for the appended-record flag that must be set on the terminating round
    before the loop is broken: the ordering is the property, not the adjacency.
    """
    lines = src.splitlines()
    idx = next((i for i, ln in enumerate(lines) if assign_needle in ln), None)
    if idx is None:
        return False
    indent = len(lines[idx]) - len(lines[idx].lstrip())
    if guard_fragment is not None:
        guards = list(_iter_enclosing(src, assign_needle, guard_fragment))
        if not guards:
            return False
    for line in lines[idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        cur = len(line) - len(line.lstrip())
        # The terminating `break` usually sits at a LOWER indent than the
        # assignment, because the assignment is inside a nested guard, so the
        # break test must come before the dedent test.
        if stripped == "break" and cur <= indent:
            return True
        if cur < indent:
            return False
    return False
