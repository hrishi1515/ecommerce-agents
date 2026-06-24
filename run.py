"""
run.py — Entry point for the full LangGraph agent pipeline
===========================================================
Usage:
    python3 run.py --input product.json --marketplace amazon
    python3 run.py --input product.json --marketplace amazon --count 2 --bullets 5 --faqs 5
    python3 run.py --text "Nike black running shoes, mesh upper, size 10"
    echo "32oz stainless bottle" | python3 run.py --stdin

Requires:
    export OPENROUTER_API_KEY=your_key_here
"""

import argparse
import json
import os
import sys

from graph import pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the full agent pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  help="Path to .json, .csv, or text file")
    group.add_argument("--text",   help="Raw product description string")
    group.add_argument("--stdin",  action="store_true", help="Read from stdin")

    parser.add_argument("--marketplace", default="generic",
                        choices=["amazon", "walmart", "etsy", "generic"])
    parser.add_argument("--count",   type=int, default=1,     help="Number of title options")
    parser.add_argument("--bullets", type=int, default=5,     help="Number of bullet points")
    parser.add_argument("--faqs",    type=int, default=5,     help="Number of FAQ pairs")
    parser.add_argument("--format",  default="plain",         choices=["plain", "html"])
    parser.add_argument("--model",   default="anthropic/claude-sonnet-4")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    raw_input = sys.stdin.read().strip() if args.stdin else (args.text or args.input)

    initial_state = {
        "raw_input":              raw_input,
        "parsed_attributes":      {},
        "cleaned_attributes":     {},
        "cleaning_changes":       [],
        "cleaning_flags":         [],
        "titles":                 [],
        "description":            "",
        "description_word_count": 0,
        "description_keywords":   [],
        "bullets":                [],
        "bullets_valid":          True,
        "bullets_issues":         [],
        "primary_keywords":       [],
        "secondary_keywords":     [],
        "tags":                   [],
        "backend_keywords":       "",
        "keywords_valid":         True,
        "faqs":                   [],
        "faq_categories":         [],
        "grammar_issues":         [],
        "grammar_clean":          True,
        "corrected_description":  "",
        "readability":            "easy",
        "meta_title":             "",
        "meta_description":       "",
        "meta_title_valid":       True,
        "meta_desc_valid":        True,
        "meta_issues":            [],
        "marketplace":            args.marketplace,
        "title_count":            args.count,
        "bullet_count":           args.bullets,
        "faq_count":              args.faqs,
        "output_format":          args.format,
        "model":                  args.model,
        "errors":                 [],
    }

    print("=" * 55)
    print("  Agent Pipeline — LangGraph")
    print("=" * 55)

    final_state = pipeline.invoke(initial_state)

    # ── Results ──────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)

    # Titles
    print(f"\n📌 TITLES ({args.marketplace.upper()}):")
    for i, r in enumerate(final_state["titles"], 1):
        status = "✓" if r["valid"] else "⚠"
        print(f"  {i}. {status} ({r['length']} chars) {r['title']}")
        for issue in r.get("issues", []):
            print(f"       ⚠ {issue}")

    # Description (corrected if grammar issues found)
    desc = final_state.get("corrected_description") or final_state["description"]
    wc   = final_state["description_word_count"]
    grammar_note = "" if final_state["grammar_clean"] else " — grammar corrected"
    print(f"\n📝 DESCRIPTION ({wc} words{grammar_note}):")
    print(f"  {desc}")
    if not final_state["grammar_clean"]:
        print(f"\n  Grammar issues fixed:")
        for issue in final_state["grammar_issues"]:
            print(f"    - {issue}")
    print(f"  Readability: {final_state['readability']}")

    # Bullets
    print(f"\n🔹 BULLET POINTS:")
    for i, bullet in enumerate(final_state["bullets"], 1):
        print(f"  {i}. {bullet}")

    # Keywords
    print(f"\n🔑 KEYWORDS:")
    print(f"  Primary   : {', '.join(final_state['primary_keywords'])}")
    print(f"  Secondary : {', '.join(final_state['secondary_keywords'])}")
    print(f"  Tags      : {', '.join(final_state['tags'])}")
    backend = final_state["backend_keywords"]
    if backend:
        valid_str = "✓ within limit" if final_state["keywords_valid"] else "⚠ over limit"
        print(f"  Backend   : {backend}")
        print(f"  ({len(backend)} chars — {valid_str})")

    # FAQs
    print(f"\n❓ FAQs (topics: {', '.join(final_state['faq_categories'])}):")
    for i, faq in enumerate(final_state["faqs"], 1):
        print(f"\n  Q{i}: {faq.get('question', '')}")
        print(f"  A{i}: {faq.get('answer', '')}")

    # Meta SEO
    t_status = "✓" if final_state["meta_title_valid"] else "⚠"
    d_status = "✓" if final_state["meta_desc_valid"]  else "⚠"
    print(f"\n🔍 META SEO:")
    print(f"  {t_status} Title ({final_state['meta_title_len'] if hasattr(final_state, 'meta_title_len') else len(final_state['meta_title'])} chars): {final_state['meta_title']}")
    print(f"  {d_status} Desc  ({len(final_state['meta_description'])} chars): {final_state['meta_description']}")
    if final_state["meta_issues"]:
        for issue in final_state["meta_issues"]:
            print(f"    ⚠ {issue}")

    # Cleaning changes
    if final_state["cleaning_changes"]:
        print(f"\n🔧 CLEANING CHANGES:")
        for c in final_state["cleaning_changes"]:
            print(f"  - {c}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()