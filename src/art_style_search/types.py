"""Shared data structures for the art style search loop."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Caption:
    """A cached caption for a single reference image."""

    image_path: Path
    text: str


@dataclass(frozen=True)
class MetricScores:
    """Evaluation scores for a single generated image against references."""

    dino_similarity: float  # higher = better, cosine sim of DINOv2 embeddings
    lpips_distance: float  # lower = better, perceptual distance
    hps_score: float  # higher = better, human preference
    aesthetics_score: float  # higher = better, 1-10 scale


@dataclass(frozen=True)
class AggregatedMetrics:
    """Mean + std of MetricScores across all generated images for an iteration."""

    dino_similarity_mean: float
    dino_similarity_std: float
    lpips_distance_mean: float
    lpips_distance_std: float
    hps_score_mean: float
    hps_score_std: float
    aesthetics_score_mean: float
    aesthetics_score_std: float

    def summary_dict(self) -> dict[str, float]:
        """Flat dict for JSON serialization and Claude consumption."""
        return {
            "dino_similarity_mean": self.dino_similarity_mean,
            "dino_similarity_std": self.dino_similarity_std,
            "lpips_distance_mean": self.lpips_distance_mean,
            "lpips_distance_std": self.lpips_distance_std,
            "hps_score_mean": self.hps_score_mean,
            "hps_score_std": self.hps_score_std,
            "aesthetics_score_mean": self.aesthetics_score_mean,
            "aesthetics_score_std": self.aesthetics_score_std,
        }


def composite_score(m: AggregatedMetrics) -> float:
    """Weighted composite score for ranking branches. Higher = better."""
    return (
        0.4 * m.dino_similarity_mean
        - 0.2 * m.lpips_distance_mean
        + 0.2 * m.hps_score_mean
        + 0.2 * (m.aesthetics_score_mean / 10.0)
    )


@dataclass(frozen=True)
class StyleProfile:
    """Structured art style analysis — foundation for all prompt work."""

    color_palette: str
    composition: str
    technique: str
    mood_atmosphere: str
    subject_matter: str
    influences: str
    gemini_raw_analysis: str
    claude_raw_analysis: str


@dataclass
class PromptSection:
    """One section of the prompt template."""

    name: str
    description: str
    value: str


@dataclass
class PromptTemplate:
    """The prompt's structure (sections) — evolves separately from content.

    The template defines WHAT sections exist and their purpose.
    Values fill the sections with specific style descriptions.
    Claude can propose changes to either the template or the values.
    """

    sections: list[PromptSection] = field(default_factory=list)
    negative_prompt: str | None = None

    def render(self) -> str:
        """Combine all section values into the final image generation prompt."""
        parts = [s.value for s in self.sections if s.value]
        if self.negative_prompt:
            parts.append(f"Avoid: {self.negative_prompt}")
        return " ".join(parts)


class ConvergenceReason(enum.Enum):
    MAX_ITERATIONS = "max_iterations"
    PLATEAU = "plateau"
    CLAUDE_STOP = "claude_stop"


@dataclass
class IterationResult:
    """Complete record of one iteration for one branch."""

    branch_id: int
    iteration: int
    template: PromptTemplate
    rendered_prompt: str
    image_paths: list[Path]
    per_image_scores: list[MetricScores]
    aggregated: AggregatedMetrics
    claude_analysis: str
    template_changes: str
    kept: bool
    vision_feedback: str = ""


@dataclass
class BranchState:
    """Mutable state for a single population branch."""

    branch_id: int
    current_template: PromptTemplate
    best_template: PromptTemplate
    best_metrics: AggregatedMetrics | None = None
    history: list[IterationResult] = field(default_factory=list)
    plateau_counter: int = 0
    stopped: bool = False
    stop_reason: ConvergenceReason | None = None


@dataclass
class LoopState:
    """Top-level state that gets persisted to state.json."""

    iteration: int
    branches: list[BranchState]
    captions: list[Caption]
    style_profile: StyleProfile
    global_best_prompt: str = ""
    global_best_metrics: AggregatedMetrics | None = None
    converged: bool = False
    convergence_reason: ConvergenceReason | None = None
