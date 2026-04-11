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
    target_category: str = ""


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
    target_category: str = ""
