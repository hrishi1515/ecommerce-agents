"""
tools/data_cleaner_tool.py
===========================
LangChain @tool — two-layer data cleaning:
  1. Rule-based : whitespace, units, casing, dedup, empty fields
  2. LLM-based  : typos, inconsistent vocab, suspicious-value flags
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Rule-based layer
# ---------------------------------------------------------------------------

UNIT_MAP = {
    r"\bounces?\b|\boz\.?\b": "oz",
    r"\bpounds?\b|\blbs?\.?\b": "lb",
    r"\binches?\b|\bin\.?\b": "in",
    r"\bcentimeters?\b|\bcm\.?\b": "cm",
    r"\bmillimeters?\b|\bmm\.?\b": "mm",
    r"\bkilograms?\b|\bkgs?\.?\b": "kg",
    r"\bgrams?\b|\bgs?\.?\b": "g",
    r"\bliters?\b|\bl\.?\b": "l",
    r"\bmilliliters?\b|\bml\.?\b": "ml",
}

TITLE_CASE_FIELDS = {"color", "material", "brand", "product_type", "category"}


def _normalize_units(value: str) -> str:
    for pattern, canonical in UNIT_MAP.items():
        value = re.sub(pattern, canonical, value, flags=re.IGNORECASE)
    return value


def _rule_clean_value(key: str, value, changes: list):
    if isinstance(value, str):
        original = value
        v = value.strip()
        v = re.sub(r"\s+", " ", v)
        v = _normalize_units(v)
        if key.lower() in TITLE_CASE_FIELDS and v:
            v = v.title()
        if v != original:
            changes.append(f"'{key}': '{original}' → '{v}' (rule)")
        return v

    if isinstance(value, list):
        original_len = len(value)
        seen, cleaned = set(), []
        for item in value:
            c = _rule_clean_value(key, item, changes) if isinstance(item, str) else item
            key_str = json.dumps(c, sort_keys=True) if isinstance(c, (dict, list)) else c
            if key_str not in seen:
                seen.add(key_str)
                cleaned.append(c)
        if len(cleaned) != original_len:
            changes.append(f"'{key}': removed {original_len - len(cleaned)} duplicate(s) (rule)")
        return cleaned

    return value


def _rule_clean(attributes: dict) -> tuple:
    changes, cleaned = [], {}
    for k, v in attributes.items():
        if v in (None, "", [], {}):
            changes.append(f"'{k}': removed (empty)")
            continue
        cleaned[k] = _rule_clean_value(k, v, changes)
    return cleaned, changes


# ---------------------------------------------------------------------------
# LLM-based layer
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are a product data normalization agent.

The JSON below has already been through rule-based cleaning. Your job:
- Fix obvious typos in values
- Standardize inconsistent naming (e.g. "SS", "stainless-steel" → "Stainless Steel")
- FLAG (do not silently fix) anything suspicious, contradictory, or likely wrong
- Do NOT add new attributes or change values that are already correct

Product attributes:
{attributes}

Return ONLY this JSON shape, no preamble, no markdown fences:
{{
  "cleaned": {{ ...corrected attributes... }},
  "changes": ["description of each change"],
  "flags":   ["description of anything suspicious NOT auto-changed"]
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://localhost", "X-Title": "Data Cleaner Agent"},
    )


def _llm_clean(attributes: dict, model: str) -> dict:
    llm = _get_llm(model)
    prompt = LLM_PROMPT.format(attributes=json.dumps(attributes, indent=2))
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "cleaned": attributes,
            "changes": [],
            "flags": [f"LLM response could not be parsed: {raw[:200]}"],
        }


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

@tool
def clean_data_tool(
    attributes: dict,
    model: str = "anthropic/claude-sonnet-4",
    skip_llm: bool = False,
) -> dict:
    """
    Cleans and normalizes product attribute data.
    Pass in a dict of product attributes.
    Returns: { cleaned: dict, changes: list, flags: list }
    """
    rule_cleaned, rule_changes = _rule_clean(attributes)

    if skip_llm:
        return {"cleaned": rule_cleaned, "changes": rule_changes, "flags": []}

    llm_result = _llm_clean(rule_cleaned, model)
    return {
        "cleaned": llm_result.get("cleaned", rule_cleaned),
        "changes": rule_changes + llm_result.get("changes", []),
        "flags":   llm_result.get("flags", []),
    }
