"""
tools/bullet_generator_tool.py
================================
LangChain @tool — generates scannable "About this item" bullet points.

Input  : cleaned product attributes dict
Output : {
    "bullets"       : list of str  — each a single bullet point
    "bullet_count"  : int
    "valid"         : bool         — True if all bullets pass length check
    "issues"        : list of str  — any bullets that are too long
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MARKETPLACE_RULES = {
    "amazon": {
        "max_chars":    500,
        "style": (
            "Amazon bullet style: start each bullet with a short ALL-CAPS keyword phrase "
            "followed by an em dash, then the benefit. "
            "Example: 'LEAK-PROOF LID — keeps drinks sealed during commutes and workouts.' "
            "Focus on customer benefit, not just feature. Max 500 chars per bullet."
        ),
    },
    "walmart": {
        "max_chars":    500,
        "style": (
            "Walmart bullet style: clear, factual, benefit-first sentences. "
            "No ALL CAPS. No hype. Max 500 chars per bullet."
        ),
    },
    "etsy": {
        "max_chars":    500,
        "style": (
            "Etsy bullet style: warm and descriptive. "
            "Each bullet highlights a feature with a personal, lifestyle angle. "
            "Max 500 chars per bullet."
        ),
    },
    "generic": {
        "max_chars":    300,
        "style": (
            "Clear, benefit-first bullet points. "
            "Start with the feature, explain why it matters to the customer. "
            "Max 300 chars per bullet."
        ),
    },
}

BANNED_WORDS = ["best", "amazing", "perfect", "#1", "guaranteed", "cheap"]

PROMPT = """You are an expert e-commerce copywriter specializing in product bullet points.

Marketplace  : {marketplace}
Style rules  : {style}
Bullet count : exactly {count} bullets

Product attributes:
{attributes}

Write exactly {count} bullet points that:
- Each focuses on ONE distinct feature/benefit (no repetition across bullets)
- Lead with the most important selling point first
- Use only information from the attributes — do not invent specs
- Avoid these words: {banned}

Return ONLY a JSON array of strings (one string per bullet), no preamble, no markdown fences.
["Bullet 1 text", "Bullet 2 text", ...]"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Bullet Generator Agent",
        },
    )


def _validate_bullets(bullets: list, marketplace: str) -> tuple:
    rules = MARKETPLACE_RULES[marketplace]
    issues = []
    for i, bullet in enumerate(bullets, 1):
        if len(bullet) > rules["max_chars"]:
            issues.append(
                f"Bullet {i} exceeds {rules['max_chars']} chars ({len(bullet)} actual)"
            )
        for word in BANNED_WORDS:
            if word in bullet.lower():
                issues.append(f"Bullet {i} contains banned word: '{word}'")
    return len(issues) == 0, issues


@tool
def generate_bullets_tool(
    attributes: dict,
    marketplace: str = "generic",
    count: int = 5,
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Generates scannable bullet points for a product listing.

    Args:
        attributes : cleaned product attribute dict
        marketplace: amazon | walmart | etsy | generic
        count      : number of bullet points to generate (default 5)
        model      : OpenRouter model id

    Returns dict with: bullets, bullet_count, valid, issues
    """
    if marketplace not in MARKETPLACE_RULES:
        marketplace = "generic"

    rules = MARKETPLACE_RULES[marketplace]
    prompt = PROMPT.format(
        marketplace=marketplace,
        style=rules["style"],
        count=count,
        attributes=json.dumps(attributes, indent=2),
        banned=", ".join(BANNED_WORDS),
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        bullets = json.loads(raw)
        if not isinstance(bullets, list):
            bullets = [raw]
    except json.JSONDecodeError:
        # Fallback: try splitting on newlines if JSON fails
        bullets = [
            line.lstrip("-•* ").strip()
            for line in raw.splitlines()
            if line.strip()
        ]

    valid, issues = _validate_bullets(bullets, marketplace)

    return {
        "bullets":      bullets,
        "bullet_count": len(bullets),
        "valid":        valid,
        "issues":       issues,
    }
