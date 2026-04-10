"""Compatibility surface for shared utilities.

The concrete implementations now live in focused modules:
``retry.py``, ``media.py``, ``taxonomy.py``, and ``reasoning_client.py``.
"""

from art_style_search.media import IMAGE_EXTENSIONS, MIME_MAP, build_ref_gen_pairs, image_to_gemini_part
from art_style_search.reasoning_client import ReasoningClient, extract_text, extract_xml_tag, stream_message
from art_style_search.retry import CircuitBreaker, async_retry, gemini_circuit_breaker
from art_style_search.taxonomy import CATEGORY_SYNONYMS

__all__ = [
    "CATEGORY_SYNONYMS",
    "IMAGE_EXTENSIONS",
    "MIME_MAP",
    "CircuitBreaker",
    "ReasoningClient",
    "async_retry",
    "build_ref_gen_pairs",
    "extract_text",
    "extract_xml_tag",
    "gemini_circuit_breaker",
    "image_to_gemini_part",
    "stream_message",
]
