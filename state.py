"""
state.py — Shared LangGraph state schema
=========================================
This is the single object that flows through every node in the graph.
Each agent reads from it and writes its results back into it.
LangGraph handles passing it between nodes automatically.
"""

from typing import TypedDict, Optional, List, Dict, Any


class ProductState(TypedDict):
    # --- Input ---
    raw_input: str                        # original raw text / file path / JSON string

    # --- After input_parser_node ---
    parsed_attributes: Dict[str, Any]     # structured product attributes

    # --- After data_cleaner_node ---
    cleaned_attributes: Dict[str, Any]    # cleaned + normalized attributes
    cleaning_changes: List[str]           # audit log of what changed
    cleaning_flags: List[str]             # fields that need human review

    # --- After title_generator_node ---
    titles: List[Dict[str, Any]]          # list of title results with validation

    # --- Config (set at start, read by all nodes) ---
    marketplace: str                      # amazon | walmart | etsy | generic
    title_count: int                      # how many title options to generate
    model: str                            # OpenRouter model id

    # --- Errors (any node can write here) ---
    errors: List[str]
