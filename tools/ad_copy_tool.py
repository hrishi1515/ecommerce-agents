"""
tools/ad_copy_tool.py
=======================
LangChain @tool — generates sponsored ad headlines and copy.

Input  : cleaned product attributes + target audience
Output : {
    "headlines"         : list  — short punchy headlines (under 30 chars each)
    "short_copy"        : str   — 1-2 sentence ad copy for small placements
    "long_copy"         : str   — 3-4 sentence ad copy for larger placements
    "call_to_action"    : str   — CTA button text e.g. "Shop Now", "Order Today"
    "target_audience"   : str   — who this ad is written for
    "headline_count"    : int
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

PLATFORM_RULES = {
    "amazon":    "Amazon Sponsored Products — focus on product benefits and value, no prices",
    "google":    "Google Shopping Ads — highlight unique selling points, include key specs",
    "facebook":  "Facebook/Instagram Ads — conversational, benefit-driven, emotional hook first",
    "generic":   "General ad copy — clear benefit, strong CTA, no hype words",
}

PROMPT = """You are an expert performance marketing copywriter.

Platform      : {platform}
Platform rules: {platform_rules}
Target audience: {audience}

Product attributes:
{attributes}

Generate ad copy that:
- Leads with the strongest customer benefit (not a feature)
- Is specific — use real numbers/specs from the attributes
- Avoids hype words: amazing, best, #1, guaranteed, perfect
- Matches the platform tone and rules above
- Includes a clear call to action

Create:
1. 5 short headlines (max 30 chars each) — punchy, keyword-rich
2. Short ad copy (1-2 sentences) — for small placements like search ads
3. Long ad copy (3-4 sentences) — for larger placements like display ads
4. Call to action text (2-4 words) — e.g. "Shop Now", "Order Today"

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "headlines"      : ["Headline 1", "Headline 2", "Headline 3", "Headline 4", "Headline 5"],
  "short_copy"     : "1-2 sentence ad copy",
  "long_copy"      : "3-4 sentence ad copy",
  "call_to_action" : "CTA text",
  "target_audience": "{audience}"
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Ad Copy Generator Agent",
        },
    )


def _validate_headlines(headlines: list) -> tuple:
    issues = []
    for i, h in enumerate(headlines, 1):
        if len(h) > 30:
            issues.append(f"Headline {i} too long ({len(h)}/30 chars): '{h}'")
    return issues


@tool
def generate_ad_copy_tool(
    attributes: dict,
    platform: str = "generic",
    audience: str = "general shoppers",
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Generates sponsored ad headlines and copy for a product.

    Args:
        attributes : cleaned product attribute dict
        platform   : amazon | google | facebook | generic
        audience   : target audience description e.g. "fitness enthusiasts aged 25-40"
        model      : OpenRouter model id

    Returns dict with: headlines, short_copy, long_copy, call_to_action,
                       target_audience, headline_count
    """
    if platform not in PLATFORM_RULES:
        platform = "generic"

    prompt = PROMPT.format(
        platform=platform,
        platform_rules=PLATFORM_RULES[platform],
        audience=audience,
        attributes=json.dumps(attributes, indent=2),
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        headlines = result.get("headlines", [])
        issues = _validate_headlines(headlines)
        return {
            "headlines":      headlines,
            "short_copy":     result.get("short_copy", ""),
            "long_copy":      result.get("long_copy", ""),
            "call_to_action": result.get("call_to_action", "Shop Now"),
            "target_audience": result.get("target_audience", audience),
            "headline_count": len(headlines),
            "issues":         issues,
        }
    except json.JSONDecodeError:
        return {
            "headlines":      [],
            "short_copy":     "",
            "long_copy":      "",
            "call_to_action": "Shop Now",
            "target_audience": audience,
            "headline_count": 0,
            "issues":         ["Could not parse ad copy response"],
        }
