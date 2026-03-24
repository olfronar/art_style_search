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
            parts.append(f"Do NOT include: {self.negative_prompt}")
        return " ".join(parts)


class ConvergenceReason(enum.Enum):
    MAX_ITERATIONS = "max_iterations"
    PLATEAU = "plateau"
    CLAUDE_STOP = "claude_stop"


# ---------------------------------------------------------------------------
# Knowledge Base — structured iteration-to-iteration learning
# ---------------------------------------------------------------------------

# Synonym map for category auto-classification
_CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "color_palette": ["color", "hue", "palette", "saturation", "tone", "gradient", "shade"],
    "composition": ["layout", "framing", "spatial", "arrangement", "perspective", "depth"],
    "technique": ["medium", "brushwork", "brushstroke", "rendering", "stroke", "paint", "watercolor"],
    "mood_atmosphere": ["mood", "atmosphere", "emotion", "feeling", "ambiance", "tone"],
    "lighting": ["light", "shadow", "illumination", "glow", "highlight", "contrast"],
    "texture": ["texture", "surface", "grain", "detail", "pattern"],
    "subject_matter": ["subject", "character", "figure", "object", "scene"],
    "background": ["background", "environment", "setting", "landscape", "sky"],
}


def classify_hypothesis(text: str, categories: list[str]) -> str:
    """Auto-classify a hypothesis into a category via keyword matching."""
    text_lower = text.lower()
    best_cat = "general"
    best_score = 0
    for cat in categories:
        score = 0
        # Match words from the category name itself
        for word in cat.replace("_", " ").split():
            if word in text_lower:
                score += 1
        # Match synonyms
        for synonym in _CATEGORY_SYNONYMS.get(cat, []):
            if synonym in text_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat if best_score > 0 else "general"


def get_category_names(template: PromptTemplate) -> list[str]:
    """Merge StyleProfile field names and PromptSection names into canonical category set."""
    cats = {"color_palette", "composition", "technique", "mood_atmosphere", "subject_matter"}
    for section in template.sections:
        cats.add(section.name)
    return sorted(cats)


@dataclass
class Hypothesis:
    """A hypothesis tested during optimization, with lineage tracking."""

    id: str  # "H1", "H2", auto-incremented
    iteration: int
    parent_id: str | None  # "H3" or None for root hypotheses
    statement: str  # from <hypothesis> tag
    experiment: str  # from <experiment> tag
    category: str  # auto-classified from text
    outcome: str  # "confirmed" | "rejected" | "partial"
    metric_delta: dict[str, float]  # {"dino": +0.02, "lpips": -0.01}
    kept: bool
    lesson: str  # confirmed/rejected/insight text


@dataclass
class OpenProblem:
    """A ranked open problem — proposed by Claude, validated by code."""

    text: str  # Claude's description
    category: str  # auto-classified
    priority: str  # "HIGH" | "MED" | "LOW" — set by code from metrics
    metric_gap: float | None = None  # DINO gap vs best category
    since_iteration: int = 0  # when first identified


@dataclass
class CategoryProgress:
    """Accumulated knowledge about one style dimension."""

    category: str
    best_dino_delta: float | None = None
    confirmed_insights: list[str] = field(default_factory=list)
    rejected_approaches: list[str] = field(default_factory=list)
    hypothesis_ids: list[str] = field(default_factory=list)


@dataclass
class KnowledgeBase:
    """Structured knowledge replacing flat research_log."""

    hypotheses: list[Hypothesis] = field(default_factory=list)
    categories: dict[str, CategoryProgress] = field(default_factory=dict)
    open_problems: list[OpenProblem] = field(default_factory=list)
    next_id: int = 1

    def add_hypothesis(
        self,
        iteration: int,
        parent_id: str | None,
        statement: str,
        experiment: str,
        category: str,
        kept: bool,
        metric_delta: dict[str, float],
        lesson: str,
        confirmed: str,
        rejected: str,
    ) -> Hypothesis:
        """Create a hypothesis, append it, and update category progress."""
        hid = f"H{self.next_id}"
        self.next_id += 1

        # Determine outcome from kept + lesson text signals
        if kept and confirmed and not rejected:
            outcome = "confirmed"
        elif not kept and rejected:
            outcome = "rejected"
        elif kept and rejected:
            outcome = "partial"
        elif not kept:
            outcome = "rejected"
        else:
            outcome = "confirmed" if kept else "rejected"

        hyp = Hypothesis(
            id=hid,
            iteration=iteration,
            parent_id=parent_id,
            statement=statement,
            experiment=experiment,
            category=category,
            outcome=outcome,
            metric_delta=metric_delta,
            kept=kept,
            lesson=lesson,
        )
        self.hypotheses.append(hyp)

        # Update category progress
        cat = self.categories.get(category)
        if cat is None:
            cat = CategoryProgress(category=category)
            self.categories[category] = cat
        cat.hypothesis_ids.append(hid)

        dino_delta = metric_delta.get("dino", 0.0)
        if outcome == "confirmed" or outcome == "partial":
            if lesson and lesson not in cat.confirmed_insights:
                cat.confirmed_insights.append(lesson)
            if cat.best_dino_delta is None or dino_delta > cat.best_dino_delta:
                cat.best_dino_delta = dino_delta
        if outcome == "rejected":
            short = statement[:120]
            if short not in cat.rejected_approaches:
                cat.rejected_approaches.append(short)

        return hyp

    def render_for_claude(self, max_words: int = 1500) -> str:
        """Produce a structured, token-efficient prompt section for Claude."""
        if not self.hypotheses:
            return ""

        parts: list[str] = []
        num_cats = len(self.categories)
        parts.append(f"## Knowledge Base ({len(self.hypotheses)} hypotheses across {num_cats} categories)\n")

        # --- Hypothesis chain (tree view) ---
        parts.append("### Hypothesis Chain")
        roots: list[Hypothesis] = []
        children: dict[str, list[Hypothesis]] = {}
        for h in self.hypotheses:
            if h.parent_id is None:
                roots.append(h)
            else:
                children.setdefault(h.parent_id, []).append(h)

        def _render_hyp(h: Hypothesis, indent: int) -> str:
            prefix = "  " * indent + ("\u2514\u2500 " if indent > 0 else "")
            builds = f", builds on {h.parent_id}" if h.parent_id else ""
            truncated = h.statement[:80] + ("..." if len(h.statement) > 80 else "")
            return (
                f'{prefix}{h.id} (iter {h.iteration}, {h.category}{builds}) \u2192 {h.outcome.upper()}: "{truncated}"'
            )

        def _render_tree(h: Hypothesis, indent: int, lines: list[str]) -> None:
            lines.append(_render_hyp(h, indent))
            for child in children.get(h.id, []):
                _render_tree(child, indent + 1, lines)

        tree_lines: list[str] = []
        for root in roots:
            _render_tree(root, 0, tree_lines)
        # Also render orphans (parent_id set but parent not found — e.g. from cross-branch reference)
        known_ids = {h.id for h in self.hypotheses}
        for h in self.hypotheses:
            if h.parent_id and h.parent_id not in known_ids and h not in roots:
                tree_lines.append(_render_hyp(h, 0))

        parts.append("\n".join(tree_lines))

        # --- Per-category status ---
        parts.append("\n### Per-Category Status")
        for cat_name in sorted(self.categories):
            cat = self.categories[cat_name]
            delta_str = f", best DINO delta: {cat.best_dino_delta:+.3f}" if cat.best_dino_delta is not None else ""
            parts.append(f"**{cat_name}** [{len(cat.hypothesis_ids)} hypotheses{delta_str}]")
            if cat.confirmed_insights:
                for insight in cat.confirmed_insights:
                    parts.append(f"  \u2713 {insight}")
            if cat.rejected_approaches:
                for rej in cat.rejected_approaches:
                    parts.append(f"  \u2717 {rej}")
            if not cat.confirmed_insights and not cat.rejected_approaches:
                parts.append("  No insights yet.")

        # --- Open problems ---
        if self.open_problems:
            parts.append("\n### Open Problems (Claude-proposed, code-validated)")
            for i, prob in enumerate(self.open_problems, 1):
                gap_str = f" \u2014 DINO gap: {prob.metric_gap:.3f}" if prob.metric_gap is not None else ""
                parts.append(f"{i}. [{prob.priority}] {prob.text}{gap_str} (since iter {prob.since_iteration})")

        # --- Do NOT repeat ---
        rejected = [(h.id, h.iteration, h.statement[:80]) for h in self.hypotheses if h.outcome == "rejected"]
        if rejected:
            parts.append("\n### Do NOT Repeat")
            for hid, it, stmt in rejected:
                parts.append(f"- {stmt} ({hid}, iter {it})")

        result = "\n".join(parts)

        # Token budget: truncate if too long
        word_count = len(result.split())
        if word_count > max_words and len(tree_lines) > 5:
            # Keep only 5 most recent hypotheses in full detail, summarize rest
            recent_ids = {h.id for h in self.hypotheses[-5:]}
            summary_lines: list[str] = []
            for line in tree_lines:
                # Check if this line is for a recent hypothesis
                if any(rid in line for rid in recent_ids):
                    summary_lines.append(line)
                else:
                    # One-line compact summary
                    summary_lines.append(line.split("\u2192")[0].rstrip() + " [...]" if "\u2192" in line else line)
            # Rebuild with summarized tree
            parts_truncated = [parts[0], "### Hypothesis Chain", "\n".join(summary_lines)]
            parts_truncated.extend(parts[2 + len(tree_lines) :] if len(parts) > 2 + len(tree_lines) else [])
            result = "\n".join(p for p in parts_truncated if p)

        return result


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
    hypothesis: str = ""
    experiment: str = ""
    vision_feedback: str = ""
    roundtrip_feedback: str = ""
    iteration_captions: list[Caption] = field(default_factory=list)


@dataclass
class BranchState:
    """Mutable state for a single population branch."""

    branch_id: int
    current_template: PromptTemplate
    best_template: PromptTemplate
    best_metrics: AggregatedMetrics | None = None
    history: list[IterationResult] = field(default_factory=list)
    research_log: str = ""  # kept for backward compat; no longer shown to Claude
    knowledge_base: KnowledgeBase = field(default_factory=KnowledgeBase)
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
    fixed_references: list[Path] = field(default_factory=list)
    global_best_prompt: str = ""
    global_best_metrics: AggregatedMetrics | None = None
    converged: bool = False
    convergence_reason: ConvergenceReason | None = None
