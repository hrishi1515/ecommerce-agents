"""
tools/qa_responder_tool.py
============================
LangChain @tool — answers buyer questions from product data.

Input  : buyer question + product attributes
Output : {
    "answer"         : str   — drafted answer
    "confidence"     : str   — "high" | "medium" | "low"
    "char_count"     : int
    "needs_support"  : bool  — True if question can't be answered from product data
    "needs_support_reason" : str
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

PROMPT = """You are a knowledgeable product specialist for an e-commerce store.

A potential buyer has asked a question about a product. Answer it using ONLY
the product information provided. Do not invent specifications or make promises
not supported by the data.

Buyer question : \"\"\"{question}\"\"\"

Product attributes:
{attributes}

Guidelines:
- Be helpful, friendly, and concise (max 80 words)
- If the answer is clearly in the product data → answer confidently (high confidence)
- If the answer can be reasonably inferred → answer with a caveat (medium confidence)
- If the data doesn't support an answer → say so honestly and suggest contacting support
- Never make up specs, warranty terms, or compatibility claims
- For questions about pricing, shipping, returns → flag for support team

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "answer"               : "the drafted answer",
  "confidence"           : "high | medium | low",
  "needs_support"        : true or false,
  "needs_support_reason" : "reason if needs_support is true, else empty string"
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Customer Q&A Responder Agent",
        },
    )


@tool
def respond_to_question_tool(
    question: str,
    attributes: dict,
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Answers a buyer question using product attribute data.

    Args:
        question   : the buyer's question text
        attributes : cleaned product attribute dict
        model      : OpenRouter model id

    Returns dict with: answer, confidence, char_count, needs_support, needs_support_reason
    """
    prompt = PROMPT.format(
        question=question.strip(),
        attributes=json.dumps(attributes, indent=2),
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)
        answer = result.get("answer", "")
        return {
            "answer":               answer,
            "confidence":           result.get("confidence", "low"),
            "char_count":           len(answer),
            "needs_support":        result.get("needs_support", False),
            "needs_support_reason": result.get("needs_support_reason", ""),
        }
    except json.JSONDecodeError:
        return {
            "answer":               "",
            "confidence":           "low",
            "char_count":           0,
            "needs_support":        True,
            "needs_support_reason": "Could not parse response",
        }
