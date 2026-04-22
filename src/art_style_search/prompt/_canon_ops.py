"""A2 diff-based canon editing.

Canon edits are currently full-block rewrites: the reasoner re-emits the entire 400-800-word
``style_foundation.value`` string even to change one sentence. That wastes tokens, muddies the
canon_edit_ledger (which stores 400-char excerpts, losing sub-sentence attribution), and
biases the optimizer toward trivial rewordings or bet-the-block rewrites.

This module provides the **pure apply layer**: ``apply_canon_ops(canon_text, ops)`` takes the
current canon string and a list of edit ops, returns the edited canon. Each op is validated
and malformed input raises ``ValueError`` — callers must fix the reasoner's output, not
silently corrupt canon text.

Supported op types:

- ``replace_sentence`` — find substring ``match`` in canon, replace with ``replace``.
  Requires both ``match`` and ``replace`` keys. If ``match`` isn't found, raises.
- ``add_sentence`` — insert ``value`` at ``where`` (``"start"`` or ``"end"``).
- ``replace_slot`` — replace entire canon with ``value``. Escape hatch for full-block
  rewrites; preserves back-compat for reasoners not yet emitting finer ops.

Reasoner-prompt changes and ledger schema updates that consume these ops live in a later
commit — this module is just the mechanical apply function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "replace_sentence": ("match", "replace"),
    "add_sentence": ("where", "value"),
    "replace_slot": ("value",),
}

_VALID_WHERE = ("start", "end")


def _check_fields(op: Mapping[str, Any], op_name: str) -> None:
    required = _REQUIRED_FIELDS[op_name]
    missing = [f for f in required if f not in op]
    if missing:
        msg = f"canon op {op_name!r} missing required field(s): {missing}"
        raise ValueError(msg)


def _apply_op(canon: str, op: Mapping[str, Any]) -> str:
    op_name = op.get("op")
    if op_name not in _REQUIRED_FIELDS:
        msg = f"unknown op {op_name!r} — expected one of {list(_REQUIRED_FIELDS)}"
        raise ValueError(msg)
    _check_fields(op, op_name)

    if op_name == "replace_sentence":
        match = op["match"]
        replace = op["replace"]
        if match not in canon:
            msg = f"replace_sentence: match {match!r} not found in canon"
            raise ValueError(msg)
        return canon.replace(match, replace, 1)

    if op_name == "add_sentence":
        where = op["where"]
        value = op["value"]
        if where not in _VALID_WHERE:
            msg = f"add_sentence: invalid where={where!r}, expected one of {_VALID_WHERE}"
            raise ValueError(msg)
        return value + canon if where == "start" else canon + value

    # replace_slot — full replace
    return op["value"]


def apply_canon_ops(canon: str, ops: Sequence[Mapping[str, Any]]) -> str:
    """Apply a sequence of canon ops to *canon*, returning the edited canon.

    Ops are applied in order — the second op sees the first op's output as its input.
    Empty ops list returns *canon* unchanged.
    """
    for op in ops:
        canon = _apply_op(canon, op)
    return canon
