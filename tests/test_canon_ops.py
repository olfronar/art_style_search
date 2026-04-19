"""Unit tests for A2 diff-based canon editing (apply_canon_ops helper).

Canon edits today are full-block rewrites: the reasoner re-emits the entire
400-800-word ``style_foundation.value`` string even when changing one sentence.
A2 introduces a diff op format so the reasoner can emit structured edits
(replace_sentence / add_sentence / replace_slot) that apply against the current canon.

The helper is pure: ``apply_canon_ops(canon_text, ops) -> str``. It validates each op and
raises ``ValueError`` on malformed input — no silent corruption of canon text.
"""

from __future__ import annotations

import pytest

from art_style_search.prompt._canon_ops import apply_canon_ops


class TestApplyCanonOps:
    def test_empty_ops_returns_canon_unchanged(self) -> None:
        canon = "How to Draw: graphite pencil on paper. Shading & Light: soft."
        assert apply_canon_ops(canon, []) == canon

    def test_replace_sentence_swaps_matching_substring(self) -> None:
        canon = "How to Draw: graphite pencil on paper. Shading & Light: soft diffuse."
        ops = [{"op": "replace_sentence", "match": "graphite pencil", "replace": "fountain pen ink"}]
        result = apply_canon_ops(canon, ops)
        assert "fountain pen ink" in result
        assert "graphite pencil" not in result
        assert "Shading & Light: soft diffuse." in result

    def test_replace_sentence_raises_when_match_not_found(self) -> None:
        """A missing match is a programming bug (reasoner emitted bad op) — fail loudly."""
        canon = "How to Draw: graphite pencil."
        ops = [{"op": "replace_sentence", "match": "watercolor wash", "replace": "ink wash"}]
        with pytest.raises(ValueError, match="match"):
            apply_canon_ops(canon, ops)

    def test_add_sentence_appends_to_end(self) -> None:
        canon = "Style Invariants: MUST use warm earth tones."
        ops = [{"op": "add_sentence", "where": "end", "value": " NEVER render halftone screens."}]
        result = apply_canon_ops(canon, ops)
        assert result.endswith("NEVER render halftone screens.")
        assert result.startswith("Style Invariants: MUST use warm earth tones.")

    def test_add_sentence_prepends_with_where_start(self) -> None:
        canon = "Style Invariants: MUST use warm earth tones."
        ops = [{"op": "add_sentence", "where": "start", "value": "NEVER render photorealistic detail. "}]
        result = apply_canon_ops(canon, ops)
        assert result.startswith("NEVER render photorealistic detail.")

    def test_replace_slot_swaps_entire_canon(self) -> None:
        """replace_slot is the full-block escape hatch — semantically identical to current
        ``style_foundation.value`` rewrite. Preserves back-compat for reasoners not yet
        emitting finer ops."""
        canon = "OLD CANON"
        ops = [{"op": "replace_slot", "value": "NEW CANON with five slots declarative style."}]
        assert apply_canon_ops(canon, ops) == "NEW CANON with five slots declarative style."

    def test_multiple_ops_apply_in_order(self) -> None:
        """Ops are applied sequentially — the second op sees the first op's output as its input."""
        canon = "How to Draw: pencil on paper."
        ops = [
            {"op": "replace_sentence", "match": "pencil", "replace": "ink"},
            {"op": "add_sentence", "where": "end", "value": " MUST use wet-on-wet blending."},
        ]
        result = apply_canon_ops(canon, ops)
        assert "ink on paper" in result
        assert result.endswith("MUST use wet-on-wet blending.")

    def test_unknown_op_raises(self) -> None:
        canon = "How to Draw: pencil."
        with pytest.raises(ValueError, match="unknown op"):
            apply_canon_ops(canon, [{"op": "rotate_canon", "angle": 90}])

    def test_missing_required_fields_raises(self) -> None:
        canon = "How to Draw: pencil."
        with pytest.raises(ValueError):
            apply_canon_ops(canon, [{"op": "replace_sentence"}])  # missing match + replace
        with pytest.raises(ValueError):
            apply_canon_ops(canon, [{"op": "add_sentence", "value": "x"}])  # missing where
        with pytest.raises(ValueError):
            apply_canon_ops(canon, [{"op": "replace_slot"}])  # missing value
