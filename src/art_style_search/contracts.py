"""Shared non-persisted workflow contracts."""

from __future__ import annotations

from dataclasses import dataclass

from art_style_search.types import PromptTemplate


@dataclass
class Lessons:
    """Structured lessons from one iteration."""

    confirmed: str = ""
    rejected: str = ""
    new_insight: str = ""


@dataclass
class ExperimentSketch:
    """Lightweight proposal idea used before full expansion."""

    hypothesis: str
    target_category: str
    failure_mechanism: str
    intervention_type: str
    direction_id: str
    direction_summary: str
    risk_level: str
    expected_primary_metric: str
    builds_on: str = ""


@dataclass
class InitialTemplateSketch:
    """Zero-step lightweight sketch — one variant of an initial meta-prompt before full expansion."""

    approach_summary: str
    emphasis: str
    instruction_style: str
    caption_length_target: int
    caption_sections: list[str]
    distinguishing_feature: str


@dataclass
class RefinementResult:
    """Complete result of a template refinement by the reasoning model."""

    template: PromptTemplate
    analysis: str
    template_changes: str
    should_stop: bool
    hypothesis: str
    experiment: str
    lessons: Lessons
    builds_on: str | None
    open_problems: list[str]
    changed_section: str = ""
    changed_sections: list[str] | None = None
    target_category: str = ""
    direction_id: str = ""
    direction_summary: str = ""
    failure_mechanism: str = ""
    intervention_type: str = ""
    risk_level: str = "targeted"
    expected_primary_metric: str = ""
    expected_tradeoff: str = ""


@dataclass
class ExperimentProposal:
    """Holds the reasoning model's proposed experiment before it's executed."""

    template: PromptTemplate
    hypothesis: str
    experiment_desc: str
    builds_on: str | None
    open_problems: list[str]
    lessons: Lessons
    analysis: str = ""
    template_changes: str = ""
    changed_section: str = ""
    changed_sections: list[str] | None = None
    target_category: str = ""
    direction_id: str = ""
    direction_summary: str = ""
    failure_mechanism: str = ""
    intervention_type: str = ""
    risk_level: str = "targeted"
    expected_primary_metric: str = ""
    expected_tradeoff: str = ""
