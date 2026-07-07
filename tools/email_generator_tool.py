"""
tools/email_generator_tool.py
===============================
LangChain @tool — generates marketing email copy from product info.

Input  : cleaned product attributes + email type + audience
Output : {
    "subject_line"    : str   — email subject
    "preview_text"    : str   — preview/preheader text (max 90 chars)
    "body"            : str   — full email body
    "cta_text"        : str   — call to action button text
    "word_count"      : int
    "subject_valid"   : bool  — subject line under 60 chars
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

EMAIL_TYPES = {
    "launch":     "New product launch email — build excitement, highlight what's new and why it matters",
    "promo":      "Promotional email — highlight a sale or limited-time offer, create urgency",
    "newsletter": "Newsletter feature — educate and inform, softer sell, value-first approach",
    "restock":    "Back-in-stock email — relief + urgency, product was wanted and now available",
    "abandoned":  "Cart abandonment email — remind, reassure, remove objections, gentle nudge",
}

SUBJECT_LIMIT  = 60
PREVIEW_LIMIT  = 90

PROMPT = """You are an expert e-commerce email copywriter.

Email type    : {email_type}
Email purpose : {email_purpose}
Target audience: {audience}

Product attributes:
{attributes}

Write a marketing email that:
- Has a compelling subject line (max {subject_limit} chars) — no clickbait, no ALL CAPS,
  no spam trigger words (free, guaranteed, act now, limited time)
- Has a preview/preheader text (max {preview_limit} chars) that complements the subject
- Has a clear structure: hook → product highlight → key benefits → call to action
- Uses a conversational, benefit-first tone
- Ends with a clear CTA button text (2-5 words)
- Body should be 120-180 words — scannable, short paragraphs

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "subject_line" : "email subject here",
  "preview_text" : "preview text here",
  "body"         : "full email body here",
  "cta_text"     : "CTA button text"
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Email Generator Agent",
        },
    )


@tool
def generate_email_tool(
    attributes: dict,
    email_type: str = "launch",
    audience: str = "existing customers",
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Generates marketing email copy for a product.

    Args:
        attributes : cleaned product attribute dict
        email_type : launch | promo | newsletter | restock | abandoned
        audience   : target audience e.g. "existing customers", "new subscribers"
        model      : OpenRouter model id

    Returns dict with: subject_line, preview_text, body, cta_text,
                       word_count, subject_valid, preview_valid
    """
    if email_type not in EMAIL_TYPES:
        email_type = "launch"

    prompt = PROMPT.format(
        email_type=email_type,
        email_purpose=EMAIL_TYPES[email_type],
        audience=audience,
        attributes=json.dumps(attributes, indent=2),
        subject_limit=SUBJECT_LIMIT,
        preview_limit=PREVIEW_LIMIT,
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        subject  = result.get("subject_line", "")
        preview  = result.get("preview_text", "")
        body     = result.get("body", "")
        cta      = result.get("cta_text", "Shop Now")

        issues = []
        if len(subject) > SUBJECT_LIMIT:
            issues.append(f"Subject too long ({len(subject)}/{SUBJECT_LIMIT} chars)")
        if len(preview) > PREVIEW_LIMIT:
            issues.append(f"Preview text too long ({len(preview)}/{PREVIEW_LIMIT} chars)")

        return {
            "subject_line":  subject,
            "preview_text":  preview,
            "body":          body,
            "cta_text":      cta,
            "word_count":    len(body.split()),
            "subject_valid": len(subject) <= SUBJECT_LIMIT,
            "preview_valid": len(preview) <= PREVIEW_LIMIT,
            "issues":        issues,
        }
    except json.JSONDecodeError:
        return {
            "subject_line":  "",
            "preview_text":  "",
            "body":          "",
            "cta_text":      "Shop Now",
            "word_count":    0,
            "subject_valid": False,
            "preview_valid": False,
            "issues":        ["Could not parse email response"],
        }
