"""
tools/faq_generator_tool.py
=============================
LangChain @tool — generates a product Q&A block from attributes.

Input  : cleaned product attributes dict
Output : {
    "faqs"       : list of {"question": str, "answer": str}
    "faq_count"  : int
    "categories" : list — topics covered (shipping, sizing, care, etc.)
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

PROMPT = """You are an expert e-commerce product specialist writing a FAQ section.

Marketplace : {marketplace}
Product attributes:
{attributes}

Generate exactly {count} frequently asked questions AND answers that:
- Cover the most common buyer concerns (compatibility, sizing, care, materials,
  warranty, usage, what's included, differences from similar products)
- Are answerable from the product attributes provided — do not invent specs
- Use a helpful, friendly tone
- Keep answers concise (2-4 sentences max)
- Prioritize questions that reduce purchase hesitation or returns

Also identify which topics/categories your FAQs cover
(e.g. "sizing", "materials", "care instructions", "compatibility", "warranty").

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "faqs": [
    {{"question": "Q1?", "answer": "A1."}},
    {{"question": "Q2?", "answer": "A2."}}
  ],
  "categories": ["topic1", "topic2"]
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "FAQ Generator Agent",
        },
    )


@tool
def generate_faq_tool(
    attributes: dict,
    marketplace: str = "generic",
    count: int = 5,
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Generates a product FAQ block from cleaned product attributes.

    Args:
        attributes : cleaned product attribute dict
        marketplace: amazon | walmart | etsy | generic
        count      : number of FAQ pairs to generate (default 5)
        model      : OpenRouter model id

    Returns dict with: faqs (list of Q&A dicts), faq_count, categories
    """
    prompt = PROMPT.format(
        marketplace=marketplace,
        attributes=json.dumps(attributes, indent=2),
        count=count,
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        faqs = result.get("faqs", [])
        categories = result.get("categories", [])
    except json.JSONDecodeError:
        faqs = []
        categories = []

    return {
        "faqs":       faqs,
        "faq_count":  len(faqs),
        "categories": categories,
    }
