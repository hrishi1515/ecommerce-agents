"""
tools/keyword_generator_tool.py
=================================
LangChain @tool — generates search tags and backend keywords.

Input  : cleaned product attributes dict
Output : {
    "primary_keywords"   : list  — high priority, high volume keywords
    "secondary_keywords" : list  — supporting/long-tail keywords
    "tags"               : list  — marketplace tags (Etsy style)
    "backend_keywords"   : str   — space-separated string for backend search fields
    "total_chars"        : int   — char count of backend_keywords
    "valid"              : bool  — within marketplace backend keyword char limit
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Backend keyword character limits per marketplace
KEYWORD_LIMITS = {
    "amazon":  249,   # Amazon backend search terms limit
    "walmart": 1000,
    "etsy":    0,     # Etsy uses tags instead of backend keywords
    "generic": 500,
}

PROMPT = """You are an expert e-commerce SEO keyword researcher.

Marketplace : {marketplace}
Product attributes:
{attributes}

Generate keyword data for this product:

1. PRIMARY KEYWORDS (5-8): highest search volume, most relevant terms a buyer would search
2. SECONDARY KEYWORDS (8-12): supporting long-tail keywords, variations, use-cases
3. TAGS (10-13): short 1-3 word tags for marketplace tag fields (important for Etsy)
4. BACKEND KEYWORDS: space-separated keyword string for backend search fields
   - No repetition of words already in the title
   - No commas, no quotes
   - Stay under {char_limit} characters total (0 = not applicable)
   - Include synonyms, alternate spellings, related terms

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "primary_keywords"   : ["keyword1", "keyword2"],
  "secondary_keywords" : ["keyword1", "keyword2"],
  "tags"               : ["tag1", "tag2"],
  "backend_keywords"   : "space separated keywords here"
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Keyword Generator Agent",
        },
    )


@tool
def generate_keywords_tool(
    attributes: dict,
    marketplace: str = "generic",
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Generates search tags and backend keywords for a product listing.

    Args:
        attributes : cleaned product attribute dict
        marketplace: amazon | walmart | etsy | generic
        model      : OpenRouter model id

    Returns dict with: primary_keywords, secondary_keywords, tags,
                       backend_keywords, total_chars, valid
    """
    if marketplace not in KEYWORD_LIMITS:
        marketplace = "generic"

    char_limit = KEYWORD_LIMITS[marketplace]
    prompt = PROMPT.format(
        marketplace=marketplace,
        attributes=json.dumps(attributes, indent=2),
        char_limit=char_limit if char_limit > 0 else "N/A",
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "primary_keywords":   [],
            "secondary_keywords": [],
            "tags":               [],
            "backend_keywords":   "",
            "total_chars":        0,
            "valid":              False,
        }

    backend = result.get("backend_keywords", "")
    total_chars = len(backend)
    valid = (char_limit == 0) or (total_chars <= char_limit)

    return {
        "primary_keywords":   result.get("primary_keywords", []),
        "secondary_keywords": result.get("secondary_keywords", []),
        "tags":               result.get("tags", []),
        "backend_keywords":   backend,
        "total_chars":        total_chars,
        "valid":              valid,
    }
