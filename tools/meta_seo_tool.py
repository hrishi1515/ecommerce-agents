"""
tools/meta_seo_tool.py
========================
LangChain @tool — generates meta title and meta description for search snippets.

Input  : cleaned product attributes + generated title/description
Output : {
    "meta_title"       : str   — optimized for search snippets (~60 chars)
    "meta_description" : str   — optimized for search snippets (~160 chars)
    "meta_title_len"   : int
    "meta_desc_len"    : int
    "title_valid"      : bool  — within 60 char limit
    "desc_valid"       : bool  — within 160 char limit
    "issues"           : list
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

META_TITLE_LIMIT = 60
META_DESC_LIMIT  = 160

PROMPT = """You are an expert SEO copywriter specializing in meta tags for e-commerce.

Generate a meta title and meta description for search engine snippets (Google, Bing).

Rules:
- Meta title    : max {title_limit} characters, include primary keyword + brand,
                  no clickbait, no ALL CAPS
- Meta description: max {desc_limit} characters, include a clear value proposition,
                  naturally include 1-2 keywords, end with a soft call to action
                  (e.g. "Shop now", "Order today", "Free returns")

Product attributes:
{attributes}

Existing title (for reference):
{existing_title}

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "meta_title"       : "your meta title here",
  "meta_description" : "your meta description here"
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Meta SEO Generator Agent",
        },
    )


@tool
def generate_meta_seo_tool(
    attributes: dict,
    existing_title: str = "",
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Generates meta title and meta description for search engine snippets.

    Args:
        attributes     : cleaned product attribute dict
        existing_title : the generated listing title (used as reference)
        model          : OpenRouter model id

    Returns dict with: meta_title, meta_description, lengths, valid flags, issues
    """
    prompt = PROMPT.format(
        title_limit=META_TITLE_LIMIT,
        desc_limit=META_DESC_LIMIT,
        attributes=json.dumps(attributes, indent=2),
        existing_title=existing_title or "N/A",
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        meta_title = result.get("meta_title", "")
        meta_desc  = result.get("meta_description", "")
    except json.JSONDecodeError:
        meta_title = ""
        meta_desc  = ""

    issues = []
    if len(meta_title) > META_TITLE_LIMIT:
        issues.append(f"Meta title too long ({len(meta_title)}/{META_TITLE_LIMIT} chars)")
    if len(meta_desc) > META_DESC_LIMIT:
        issues.append(f"Meta description too long ({len(meta_desc)}/{META_DESC_LIMIT} chars)")
    if not meta_title:
        issues.append("Meta title is empty")
    if not meta_desc:
        issues.append("Meta description is empty")

    return {
        "meta_title":       meta_title,
        "meta_description": meta_desc,
        "meta_title_len":   len(meta_title),
        "meta_desc_len":    len(meta_desc),
        "title_valid":      len(meta_title) <= META_TITLE_LIMIT and bool(meta_title),
        "desc_valid":       len(meta_desc) <= META_DESC_LIMIT and bool(meta_desc),
        "issues":           issues,
    }
