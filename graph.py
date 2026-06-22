"""
graph.py — LangGraph pipeline
================================
Full pipeline:

    START
      │
      ▼
  [parse_input]
      │
      ▼
  [clean_data]
      │
      ├── has flags? ──► [flag_review]
      │                       │
      └───────────────────────┘
      │
      ▼
  [generate_title]
      │
      ▼
  [write_description]
      │
      ▼
  [generate_bullets]
      │
      ▼
    END
"""

from langgraph.graph import StateGraph, END
from state import ProductState
from tools.input_parser_tool import parse_input_tool
from tools.data_cleaner_tool import clean_data_tool
from tools.title_generator_tool import generate_title_tool
from tools.description_writer_tool import write_description_tool
from tools.bullet_generator_tool import generate_bullets_tool


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_input_node(state: ProductState) -> dict:
    print("\n[1/5] Parsing input...")
    result = parse_input_tool.invoke({
        "raw_input": state["raw_input"],
        "model": state["model"],
    })
    print(f"      ✓ Extracted {len(result)} attributes")
    return {"parsed_attributes": result}


def clean_data_node(state: ProductState) -> dict:
    print("\n[2/5] Cleaning data...")
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
    print("\n⚠ DATA FLAGS — review before publishing:")
    for flag in state["cleaning_flags"]:
        print(f"   • {flag}")
    return {}


def generate_title_node(state: ProductState) -> dict:
    print(f"\n[3/5] Generating {state['title_count']} title(s) for '{state['marketplace']}'...")
    results = generate_title_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "count": state["title_count"],
        "model": state["model"],
    })
    ok = sum(1 for r in results if r["valid"])
    print(f"      ✓ Generated {len(results)} title(s) — {ok} valid")
    return {"titles": results}


def write_description_node(state: ProductState) -> dict:
    print(f"\n[4/5] Writing description ({state['output_format']} format)...")
    result = write_description_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "output_format": state["output_format"],
        "model": state["model"],
    })
    print(f"      ✓ {result['word_count']} words, {len(result['keywords_used'])} keywords used")
    return {
        "description":          result["description"],
        "description_word_count": result["word_count"],
        "description_keywords": result["keywords_used"],
    }


def generate_bullets_node(state: ProductState) -> dict:
    print(f"\n[5/5] Generating {state['bullet_count']} bullet points...")
    result = generate_bullets_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "count": state["bullet_count"],
        "model": state["model"],
    })
    status = "✓ all valid" if result["valid"] else f"⚠ {len(result['issues'])} issue(s)"
    print(f"      {status} — {result['bullet_count']} bullets generated")
    return {
        "bullets":        result["bullets"],
        "bullets_valid":  result["valid"],
        "bullets_issues": result["issues"],
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_flag(state: ProductState) -> str:
    return "flag" if state.get("cleaning_flags") else "title"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ProductState)

    graph.add_node("parse_input",       parse_input_node)
    graph.add_node("clean_data",        clean_data_node)
    graph.add_node("flag_review",       flag_review_node)
    graph.add_node("generate_title",    generate_title_node)
    graph.add_node("write_description", write_description_node)
    graph.add_node("generate_bullets",  generate_bullets_node)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "clean_data")
    graph.add_conditional_edges(
        "clean_data",
        should_flag,
        {"flag": "flag_review", "title": "generate_title"},
    )
    graph.add_edge("flag_review",       "generate_title")
    graph.add_edge("generate_title",    "write_description")
    graph.add_edge("write_description", "generate_bullets")
    graph.add_edge("generate_bullets",  END)

    return graph.compile()


pipeline = build_graph()