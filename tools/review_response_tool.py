"""
tools/review_response_tool.py
===============================
LangChain @tool — drafts responses to customer reviews (positive, negative, neutral).

Input  : review text + rating + product attributes
Output : {
    "response"       : str   — drafted reply
    "tone"           : str   — "thank" | "resolve" | "neutral"
    "sentiment"      : str   — "positive" | "negative" | "neutral"
    "char_count"     : int
    "escalate"       : bool  — True if issue needs human escalation
    "escalate_reason": str
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

PROMPT = """You are a professional customer service specialist for an e-commerce brand.

A customer has left a review. Draft a response that is:
- Personalized — reference something specific from their review
- On-brand — professional, warm, and helpful
- Concise — max 100 words
- Actionable — if there's a complaint, offer a clear next step

Review rating : {rating}/5
Review text   : \"\"\"{review}\"\"\"

Product       : {product_name}

Guidelines:
- For 4-5 star reviews: thank sincerely, highlight what they loved, invite them back
- For 1-2 star reviews: apologize, acknowledge the issue, offer a resolution path
- For 3 star reviews: thank them, address the concern, show you value feedback
- Never be defensive or dismissive
- Never offer specific discounts or refund amounts (leave that to the support team)
- If the review mentions safety issues, legal threats, or hate speech → flag for escalation

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "response"        : "the drafted reply",
  "tone"            : "thank | resolve | neutral",
  "sentiment"       : "positive | negative | neutral",
  "escalate"        : true or false,
  "escalate_reason" : "reason if escalate is true, else empty string"
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Review Response Agent",
        },
    )


@tool
def respond_to_review_tool(
    review: str,
    rating: int,
    product_name: str = "our product",
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Drafts a professional response to a customer review.

    Args:
        review       : the customer review text
        rating       : star rating 1-5
        product_name : name of the product being reviewed
        model        : OpenRouter model id

    Returns dict with: response, tone, sentiment, char_count, escalate, escalate_reason
    """
    prompt = PROMPT.format(
        rating=rating,
        review=review.strip(),
        product_name=product_name,
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        reply = result.get("response", "")
        return {
            "response":        reply,
            "tone":            result.get("tone", "neutral"),
            "sentiment":       result.get("sentiment", "neutral"),
            "char_count":      len(reply),
            "escalate":        result.get("escalate", False),
            "escalate_reason": result.get("escalate_reason", ""),
        }
    except json.JSONDecodeError:
        return {
            "response":        "",
            "tone":            "neutral",
            "sentiment":       "neutral",
            "char_count":      0,
            "escalate":        False,
            "escalate_reason": "Could not parse response",
        }
