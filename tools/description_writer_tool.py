"""
tools/description_writer_tool.py
==================================
LangChain @tool — generates a persuasive, SEO-optimized product description.

Input  : cleaned product attributes dict
Output : {
    "description"      : str   — full HTML or plain text description
    "word_count"       : int
    "keywords_used"    : list  — keywords naturally woven in
    "format"           : str   — "html" or "plain"
}
"""

import json
import os
import re
from typing import List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MARKETPLACE_STYLE = {
    "amazon": (
        "Write for Amazon. Use short paragraphs (2-3 sentences each). "
        "Lead with the strongest benefit. Naturally include keywords. "
        "Avoid promotional fluff (best, amazing, #1). No HTML tags."
    ),
    "walmart": (
        "Write for Walmart. Clear, factual, benefit-driven. "
        "Short sentences. No hype or superlatives. Plain text only."
    ),
    "etsy": (
        "Write for Etsy. Warm, lifestyle-oriented tone. "
        "Tell a story around the product. Who made it, who it's for, "
        "why it's special. Plain text, conversational."
    ),
    "generic": (
        "Write a clear, persuasive product description. "
        "Lead with the top benefit. Cover key features and use cases. "
        "Use natural keywords. Plain text, 3-4 short paragraphs."
    ),
}

PROMPT = """You are an expert e-commerce copywriter.

Marketplace   : {marketplace}
Style rules   : {style}
Target length : {min_words}–{max_words} words
Format        : {format}

Product attributes:
{attributes}

Write a product description that:
- Leads with the strongest customer benefit (not a feature)
- Covers key features naturally woven into benefit statements
- Includes these keywords naturally (do not stuff): {keywords}
- Uses only information from the attributes — do not invent specs
- Matches the marketplace style rules above
{html_instruction}

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "description"   : "the full description text",
  "keywords_used" : ["keyword1", "keyword2"]
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Description Writer Agent",
        },
    )


def _extract_keywords(attributes: dict) -> List[str]:
    """Pull likely search keywords from attributes automatically."""
    keywords = []
    for key in ["product_type", "brand", "material", "color", "size", "key_features"]:
        val = attributes.get(key)
        if isinstance(val, list):
            keywords.extend([str(v) for v in val[:3]])
        elif val:
            keywords.append(str(val))
    return list(dict.fromkeys(keywords))  # dedupe, preserve order


@tool
def write_description_tool(
    attributes: dict,
    marketplace: str = "generic",
    output_format: str = "plain",
    min_words: int = 80,
    max_words: int = 150,
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Generates a persuasive, SEO-optimized product description.

    Args:
        attributes   : cleaned product attribute dict
        marketplace  : amazon | walmart | etsy | generic
        output_format: plain | html
        min_words    : minimum word count (default 80)
        max_words    : maximum word count (default 150)
        model        : OpenRouter model id

    Returns dict with: description, word_count, keywords_used, format
    """
    if marketplace not in MARKETPLACE_STYLE:
        marketplace = "generic"

    keywords = _extract_keywords(attributes)
    html_instruction = (
        "Format using basic HTML: <p> for paragraphs, <strong> for key terms."
        if output_format == "html"
        else "Plain text only — no HTML tags."
    )

    prompt = PROMPT.format(
        marketplace=marketplace,
        style=MARKETPLACE_STYLE[marketplace],
        min_words=min_words,
        max_words=max_words,
        format=output_format,
        attributes=json.dumps(attributes, indent=2),
        keywords=", ".join(keywords),
        html_instruction=html_instruction,
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        description = result.get("description", raw)
        keywords_used = result.get("keywords_used", [])
    except json.JSONDecodeError:
        description = raw
        keywords_used = []

    word_count = len(re.sub(r"<[^>]+>", "", description).split())

    return {
        "description":   description,
        "word_count":    word_count,
        "keywords_used": keywords_used,
        "format":        output_format,
    }
