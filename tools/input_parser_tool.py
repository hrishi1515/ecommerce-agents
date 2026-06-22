"""
tools/input_parser_tool.py
===========================
LangChain @tool that accepts any input type and returns structured product attributes.
Supports: JSON string, plain text, file path (.json / .csv / .txt)
"""

import csv
import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def _get_llm(model: str) -> ChatOpenAI:
    """Returns a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Input Parser Agent",
        },
    )


EXTRACT_PROMPT = """You are a product data extraction agent.

Extract all product attributes from the text below and return them as a flat
JSON object. Use consistent lowercase snake_case keys (e.g. brand, color,
size, material, product_type, key_features, quantity, weight).

For list-type attributes use a JSON array.
Omit keys you cannot find. Do NOT invent information not in the input.

Input text:
\"\"\"{text}\"\"\"

Return ONLY the JSON object, no preamble, no markdown fences."""


def _extract_from_text(text: str, model: str) -> dict:
    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=EXTRACT_PROMPT.format(text=text))])
    raw = re.sub(r"```json|```", "", response.content).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return {"raw_text": text}


def _load_json(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else data[0]


def _load_csv_first_row(path: str) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return {k: v for k, v in row.items() if v.strip()}


@tool
def parse_input_tool(raw_input: str, model: str = "anthropic/claude-sonnet-4") -> dict:
    """
    Parses any product input into a structured attribute dict.
    Accepts: JSON string, plain text string, or a file path (.json / .csv / .txt).
    Returns a dict of product attributes.
    """
    text = raw_input.strip()

    # 1. Is it a file path?
    if os.path.exists(text):
        ext = os.path.splitext(text)[1].lower()
        if ext == ".json":
            return _load_json(text)
        if ext == ".csv":
            return _load_csv_first_row(text)
        # text file → LLM extraction
        with open(text) as f:
            content = f.read()
        return _extract_from_text(content, model)

    # 2. Is it a JSON string?
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3. Free text → LLM extraction
    return _extract_from_text(text, model)
