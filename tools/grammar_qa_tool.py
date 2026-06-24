"""
tools/grammar_qa_tool.py
==========================
LangChain @tool — checks and corrects grammar, spelling, and readability
across all generated content (title, description, bullets, FAQs).

Input  : any text content (title, description, bullets, etc.)
Output : {
    "corrected_text" : str   — fixed version of the input
    "issues_found"   : list  — list of specific issues detected
    "issue_count"    : int
    "clean"          : bool  — True if no issues found
    "readability"    : str   — "easy" | "moderate" | "complex"
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

PROMPT = """You are an expert proofreader and copy editor for e-commerce content.

Review the text below for:
- Spelling mistakes
- Grammar errors
- Punctuation issues
- Awkward or unclear phrasing
- Inconsistent capitalization
- Redundant or repeated words

Also rate the overall readability:
- "easy"     → clear, simple sentences, any buyer can understand
- "moderate" → some complex phrasing but generally clear
- "complex"  → too technical or wordy for a typical buyer

Content type : {content_type}
Text to review:
\"\"\"
{text}
\"\"\"

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "corrected_text" : "the fully corrected version of the text",
  "issues_found"   : ["specific issue 1", "specific issue 2"],
  "readability"    : "easy | moderate | complex"
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Grammar QA Agent",
        },
    )


@tool
def grammar_qa_tool(
    text: str,
    content_type: str = "product content",
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Checks and corrects grammar, spelling, and readability of any product content.

    Args:
        text         : the content to review (title, description, bullets, etc.)
        content_type : label for context e.g. "title", "description", "bullets"
        model        : OpenRouter model id

    Returns dict with: corrected_text, issues_found, issue_count, clean, readability
    """
    if not text or not text.strip():
        return {
            "corrected_text": text,
            "issues_found":   [],
            "issue_count":    0,
            "clean":          True,
            "readability":    "easy",
        }

    prompt = PROMPT.format(content_type=content_type, text=text.strip())
    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        issues = result.get("issues_found", [])
        return {
            "corrected_text": result.get("corrected_text", text),
            "issues_found":   issues,
            "issue_count":    len(issues),
            "clean":          len(issues) == 0,
            "readability":    result.get("readability", "easy"),
        }
    except json.JSONDecodeError:
        return {
            "corrected_text": text,
            "issues_found":   ["Could not parse QA response"],
            "issue_count":    1,
            "clean":          False,
            "readability":    "unknown",
        }
