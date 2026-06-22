"""
graph.py — LangGraph pipeline
===============================
Wires the three agents as graph nodes with conditional routing:

    START
      │
      ▼
  [parse_input]          → extracts structured attributes from any input
      │
      ▼
  [clean_data]           → normalizes and validates the attributes
      │
      ├── has flags? ──► [flag_review]   → prints warnings, continues anyway
      │                       │
      └───────────────────────┘
      │
      ▼
  [generate_title]       → produces marketplace-optimized titles
      │
      ▼
    END

Each node reads from ProductState and writes its results back.
LangGraph handles passing state between nodes automatically.
"""

from langgraph.graph import StateGraph, END
from state import ProductState
from tools.input_parser_tool import parse_input_tool
from tools.data_cleaner_tool import clean_data_tool
from tools.title_generator_tool import generate_title_tool


# ---------------------------------------------------------------------------
# Nodes — each receives the full state dict, returns partial updates
# ---------------------------------------------------------------------------

def parse_input_node(state: ProductState) -> dict:
    """Node 1: Parse any input format into structured attributes."""
    print("\n[1/3] Parsing input...")
    result = parse_input_tool.invoke({
        "raw_input": state["raw_input"],
        "model": state["model"],
    })
    print(f"      ✓ Extracted {len(result)} attributes")
    return {"parsed_attributes": result}


def clean_data_node(state: ProductState) -> dict:
    """Node 2: Clean and normalize the parsed attributes."""
    print("\n[2/3] Cleaning data...")
    result = clean_data_tool.invoke({
        "attributes": state["parsed_attributes"],
        "model": state["model"],
    })
    if result["changes"]:
        print(f"      ✓ Applied {len(result['changes'])} fix(es)")
    if result["flags"]:
        print(f"      ⚠ {len(result['flags'])} flag(s) need review")
    return {
        "cleaned_attributes": result["cleaned"],
        "cleaning_changes":   result["changes"],
        "cleaning_flags":     result["flags"],
    }


def flag_review_node(state: ProductState) -> dict:
    """Node 3a (conditional): Surface flags for human review."""
    print("\n⚠ DATA FLAGS — review before publishing:")
    for flag in state["cleaning_flags"]:
        print(f"   • {flag}")
    return {}   # no state changes, just a warning gate


def generate_title_node(state: ProductState) -> dict:
    """Node 3b: Generate marketplace-optimized titles."""
    print(f"\n[3/3] Generating {state['title_count']} title(s) for '{state['marketplace']}'...")
    results = generate_title_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "count": state["title_count"],
        "model": state["model"],
    })
    ok = sum(1 for r in results if r["valid"])
    print(f"      ✓ Generated {len(results)} title(s) — {ok} valid")
    return {"titles": results}


# ---------------------------------------------------------------------------
# Conditional edge — route through flag_review only when flags exist
# ---------------------------------------------------------------------------

def should_flag(state: ProductState) -> str:
    """Return 'flag' if there are cleaning flags, else go straight to titles."""
    return "flag" if state.get("cleaning_flags") else "title"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ProductState)

    # Register nodes
    graph.add_node("parse_input",     parse_input_node)
    graph.add_node("clean_data",      clean_data_node)
    graph.add_node("flag_review",     flag_review_node)
    graph.add_node("generate_title",  generate_title_node)

    # Edges
    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "clean_data")

    # Conditional: if data has flags → show review node first
    graph.add_conditional_edges(
        "clean_data",
        should_flag,
        {
            "flag":  "flag_review",
            "title": "generate_title",
        },
    )

    graph.add_edge("flag_review",    "generate_title")
    graph.add_edge("generate_title", END)

    return graph.compile()


# Singleton — import this in run.py
pipeline = build_graph()
