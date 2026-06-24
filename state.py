"""
state.py — Shared LangGraph state schema
"""
from typing import TypedDict, List, Dict, Any


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

    # --- After generate_keywords_node ---
    primary_keywords: List[str]
    secondary_keywords: List[str]
    tags: List[str]
    backend_keywords: str
    keywords_valid: bool

    # --- After generate_faq_node ---
    faqs: List[Dict[str, str]]
    faq_categories: List[str]

    # --- After grammar_qa_node ---
    grammar_issues: List[str]
    grammar_clean: bool
    corrected_description: str
    readability: str

    # --- After meta_seo_node ---
    meta_title: str
    meta_description: str
    meta_title_valid: bool
    meta_desc_valid: bool
    meta_issues: List[str]

    # --- Config ---
    marketplace: str
    title_count: int
    bullet_count: int
    faq_count: int
    output_format: str
    model: str

    # --- Errors ---
    errors: List[str]