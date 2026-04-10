"""Shared hypothesis taxonomy and category synonyms."""

from __future__ import annotations

# Canonical hypothesis categories used for classification, diversity enforcement,
# and target-category ranking.
CATEGORY_SYNONYMS: dict[str, list[str]] = {
    "color_palette": ["color", "hue", "palette", "saturation", "tone", "gradient", "shade"],
    "composition": ["layout", "framing", "spatial", "arrangement", "perspective", "depth"],
    "technique": ["medium", "brushwork", "brushstroke", "rendering", "stroke", "paint", "watercolor"],
    "mood_atmosphere": ["mood", "atmosphere", "emotion", "feeling", "ambiance", "tone"],
    "lighting": ["light", "shadow", "illumination", "glow", "highlight", "contrast"],
    "texture": ["texture", "surface", "grain", "detail", "pattern"],
    "subject_matter": ["subject", "character", "figure", "object", "scene"],
    "background": ["background", "environment", "setting", "landscape", "sky"],
    "caption_structure": ["section", "label", "order", "ordering", "structure", "format", "length"],
}
