"""Reasoning-model API surface for meta-prompt optimization.

The package is organised as:

- ``_format`` — rendering dataclasses into text blocks
- ``_parse`` — ``validate_template`` structural invariants
- ``json_contracts`` — JSON payload validators used by every per-iteration call
- ``initial`` — zero-step: propose N diverse initial templates
- ``experiments`` — per-iteration: propose N experiment branches, dedup by category
- ``synthesis`` — merge top experiments' best sections into one template
- ``review`` — CycleResearcher-inspired independent review pass

All public symbols are re-exported here so existing callers can continue to
``from art_style_search.prompt import X``.
"""

from art_style_search.contracts import ExperimentSketch, InitialTemplateSketch, Lessons, RefinementResult
from art_style_search.prompt._format import _format_metrics, _format_style_profile, _format_template
from art_style_search.prompt._parse import validate_template
from art_style_search.prompt.experiments import (
    brainstorm_experiment_sketches,
    enforce_hypothesis_diversity,
    expand_experiment_sketches,
    propose_experiments,
    rank_experiment_sketches,
    select_experiment_portfolio,
)
from art_style_search.prompt.initial import (
    brainstorm_initial_sketches,
    expand_initial_sketches,
    propose_initial_templates,
    rank_initial_sketches,
)
from art_style_search.prompt.review import review_iteration
from art_style_search.prompt.synthesis import synthesize_templates

__all__ = [
    "ExperimentSketch",
    "InitialTemplateSketch",
    "Lessons",
    "RefinementResult",
    "_format_metrics",
    "_format_style_profile",
    "_format_template",
    "brainstorm_experiment_sketches",
    "brainstorm_initial_sketches",
    "enforce_hypothesis_diversity",
    "expand_experiment_sketches",
    "expand_initial_sketches",
    "propose_experiments",
    "propose_initial_templates",
    "rank_experiment_sketches",
    "rank_initial_sketches",
    "review_iteration",
    "select_experiment_portfolio",
    "synthesize_templates",
    "validate_template",
]
