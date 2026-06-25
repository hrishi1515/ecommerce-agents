"""
tools/brand_voice_tool.py
===========================
LangChain @tool — checks if content matches brand style guide and rewrites if needed.

Input  : content text + brand style guide description
Output : {
    "on_brand"       : bool
    "score"          : int   — 0-100 brand alignment score
    "issues"         : list  — specific off-brand phrases/tone issues
    "rewritten"      : str   — on-brand rewrite of the content
    "suggestions"    : list  — tips to improve brand alignment
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

DEFAULT_BRAND_GUIDE = (
    "Professional yet approachable tone. Clear and benefit-focused. "
    "No slang, no hype words, no exclamation marks. "
    "Sentences should be concise and easy to understand."
)

PROMPT = """You are a brand voice specialist and copy editor.

Review the content below against the brand style guide provided.
Check for tone, vocabulary, sentence structure, and overall alignment.

Brand Style Guide:
\"\"\"{brand_guide}\"\"\"

Content to review ({content_type}):
\"\"\"{content}\"\"\"

Tasks:
1. Score brand alignment 0-100 (100 = perfectly on brand)
2. List specific issues (off-brand words, wrong tone, style violations)
3. Rewrite the content to be fully on-brand (keep the same meaning and facts)
4. Give 2-3 actionable suggestions for future content

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "on_brand"    : true or false,
  "score"       : 0-100,
  "issues"      : ["issue 1", "issue 2"],
  "rewritten"   : "the on-brand rewritten version",
  "suggestions" : ["tip 1", "tip 2"]
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Brand Voice Checker Agent",
        },
    )


@tool
def check_brand_voice_tool(
    content: str,
    brand_guide: str = DEFAULT_BRAND_GUIDE,
    content_type: str = "product description",
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Checks content against a brand style guide and rewrites if off-brand.

    Args:
        content      : the text to review (description, bullets, title, etc.)
        brand_guide  : your brand style guide as a text description
        content_type : label e.g. "description", "title", "bullets"
        model        : OpenRouter model id

    Returns dict with: on_brand, score, issues, rewritten, suggestions
    """
    if not content or not content.strip():
        return {
            "on_brand":    True,
            "score":       100,
            "issues":      [],
            "rewritten":   content,
            "suggestions": [],
        }

    prompt = PROMPT.format(
        brand_guide=brand_guide,
        content_type=content_type,
        content=content.strip(),
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        return {
            "on_brand":    result.get("on_brand", True),
            "score":       result.get("score", 100),
            "issues":      result.get("issues", []),
            "rewritten":   result.get("rewritten", content),
            "suggestions": result.get("suggestions", []),
        }
    except json.JSONDecodeError:
        return {
            "on_brand":    False,
            "score":       0,
            "issues":      ["Could not parse brand voice response"],
            "rewritten":   content,
            "suggestions": [],
        }
