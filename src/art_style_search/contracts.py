"""Shared non-persisted workflow contracts."""

from __future__ import annotations

from dataclasses import dataclass

from art_style_search.prompt import Lessons
from art_style_search.types import PromptTemplate


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
