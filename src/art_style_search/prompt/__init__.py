"""Reasoning-model API surface for meta-prompt optimization.

The package is organised as:

- ``_format`` — rendering dataclasses into text blocks
- ``_parse`` — regexes and parsing helpers, plus ``Lessons`` and ``RefinementResult``
- ``initial`` — zero-step: propose N diverse initial templates
- ``experiments`` — per-iteration: propose N experiment branches, dedup by category
- ``synthesis`` — merge top experiments' best sections into one template
- ``review`` — CycleResearcher-inspired independent review pass

All public symbols are re-exported here so existing callers can continue to
``from art_style_search.prompt import X``.
"""

from art_style_search.contracts import ExperimentSketch
from art_style_search.prompt._format import _format_metrics, _format_style_profile, _format_template
from art_style_search.prompt._parse import (
    Lessons,
    RefinementResult,
    _parse_analysis,
    _parse_builds_on,
    _parse_changed_section,
    _parse_converged,
    _parse_experiment,
    _parse_hypothesis,
    _parse_initial_templates,
    _parse_lessons,
    _parse_open_problems,
    _parse_refinement_branches,
    _parse_target_category,
    _parse_template,
    _parse_template_changes,
    validate_template,
)
from art_style_search.prompt.experiments import (
    brainstorm_experiment_sketches,
    enforce_hypothesis_diversity,
    expand_experiment_sketches,
    propose_experiments,
    rank_experiment_sketches,
    select_experiment_portfolio,
)
from art_style_search.prompt.initial import propose_initial_templates
from art_style_search.prompt.review import review_iteration
from art_style_search.prompt.synthesis import synthesize_templates

__all__ = [
    "ExperimentSketch",
    "Lessons",
    "RefinementResult",
    "_format_metrics",
    "_format_style_profile",
    "_format_template",
    "_parse_analysis",
    "_parse_builds_on",
    "_parse_changed_section",
    "_parse_converged",
    "_parse_experiment",
    "_parse_hypothesis",
    "_parse_initial_templates",
    "_parse_lessons",
    "_parse_open_problems",
    "_parse_refinement_branches",
    "_parse_target_category",
    "_parse_template",
    "_parse_template_changes",
    "brainstorm_experiment_sketches",
    "enforce_hypothesis_diversity",
    "expand_experiment_sketches",
    "propose_experiments",
    "propose_initial_templates",
    "rank_experiment_sketches",
    "review_iteration",
    "select_experiment_portfolio",
    "synthesize_templates",
    "validate_template",
]
