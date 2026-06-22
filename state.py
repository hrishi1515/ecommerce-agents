"""
state.py — Shared LangGraph state schema
"""
from typing import TypedDict, Optional, List, Dict, Any


class ProductState(TypedDict):
    # --- Input ---
    raw_input: str

    # --- After parse_input_node ---
    parsed_attributes: Dict[str, Any]

    # --- After clean_data_node ---
    cleaned_attributes: Dict[str, Any]
    cleaning_changes: List[str]
    cleaning_flags: List[str]

    # --- After generate_title_node ---
    titles: List[Dict[str, Any]]

    # --- After write_description_node ---
    description: str
    description_word_count: int
    description_keywords: List[str]

    # --- After generate_bullets_node ---
    bullets: List[str]
    bullets_valid: bool
    bullets_issues: List[str]

    # --- Config ---
    marketplace: str
    title_count: int
    bullet_count: int
    output_format: str       # plain | html
    model: str

    # --- Errors ---
    errors: List[str]