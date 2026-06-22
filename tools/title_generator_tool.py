"""
tools/title_generator_tool.py
==============================
LangChain @tool — generates marketplace-optimized product titles.
"""

import json
import os
import re
from typing import List

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MARKETPLACE_RULES = {
    "amazon": {
        "max_length": 200,
        "style": (
            "Brand + Product Type + Key Features + Size/Color/Quantity. "
            "No promotional words. No ALL CAPS. No emojis or special symbols."
        ),
    },
    "walmart": {
        "max_length": 100,
        "style": "Brand + Product Name + Attribute + Size/Color/Pack Count. Title case, no symbols.",
    },
    "etsy": {
        "max_length": 140,
        "style": "Keyword-rich, slightly lifestyle tone. Front-load top keywords in first 40 chars.",
    },
    "generic": {
        "max_length": 150,
        "style": "Clear, descriptive, front-loaded with keywords, no promotional fluff.",
    },
}

BANNED_WORDS = ["best", "cheap", "free shipping", "sale", "#1", "guarantee", "100%", "amazing", "perfect"]

PROMPT = """You are a product title generator for e-commerce listings.

Marketplace : {marketplace}
Max length  : {max_length} characters
Style rules : {style}

Product attributes:
{attributes}

Generate {count} title option(s):
- Respect the {max_length}-char limit strictly
- Lead with the most important search keywords
- Use only information in the attributes — do not invent specs
- Avoid these words: {banned}

Return ONLY a JSON array of strings, no preamble, no markdown fences.
["Title 1", "Title 2"]"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://localhost", "X-Title": "Title Generator Agent"},
    )


def _validate(title: str, marketplace: str) -> dict:
    rules = MARKETPLACE_RULES[marketplace]
    issues = []
    if len(title) > rules["max_length"]:
        issues.append(f"Exceeds {rules['max_length']} chars ({len(title)} actual)")
    for word in BANNED_WORDS:
        if word in title.lower():
            issues.append(f"Banned word: '{word}'")
    if title.isupper():
        issues.append("ALL CAPS")
    return {"title": title, "length": len(title), "valid": not issues, "issues": issues}


@tool
def generate_title_tool(
    attributes: dict,
    marketplace: str = "generic",
    count: int = 1,
    model: str = "anthropic/claude-sonnet-4",
) -> List[dict]:
    """
    Generates marketplace-optimized product titles from a cleaned attribute dict.
    Returns a list of title results, each with: title, length, valid, issues.
    marketplace options: amazon, walmart, etsy, generic
    """
    if marketplace not in MARKETPLACE_RULES:
        raise ValueError(f"Unknown marketplace '{marketplace}'. Choose: {list(MARKETPLACE_RULES)}")

    rules = MARKETPLACE_RULES[marketplace]
    llm = _get_llm(model)
    prompt = PROMPT.format(
        marketplace=marketplace,
        max_length=rules["max_length"],
        style=rules["style"],
        attributes=json.dumps(attributes, indent=2),
        count=count,
        banned=", ".join(BANNED_WORDS),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        titles = json.loads(raw)
        if not isinstance(titles, list):
            titles = [raw]
    except json.JSONDecodeError:
        titles = [raw]

    return [_validate(t, marketplace) for t in titles]
