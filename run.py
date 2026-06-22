"""
run.py — Entry point for the full LangGraph agent pipeline
===========================================================
Usage:
    python3 run.py --input product.json --marketplace amazon
    python3 run.py --input product.json --marketplace amazon --count 3 --bullets 5
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
    parser.add_argument("--count",   type=int, default=1,      help="Number of title options")
    parser.add_argument("--bullets", type=int, default=5,      help="Number of bullet points")
    parser.add_argument("--format",  default="plain",          choices=["plain", "html"],
                        help="Description output format")
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
        "marketplace":            args.marketplace,
        "title_count":            args.count,
        "bullet_count":           args.bullets,
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
    print(f"\n📌 TITLES ({args.marketplace}):")
    for i, r in enumerate(final_state["titles"], 1):
        status = "✓" if r["valid"] else "⚠"
        print(f"  {i}. {status} ({r['length']} chars) {r['title']}")
        for issue in r.get("issues", []):
            print(f"       ⚠ {issue}")

    # Description
    print(f"\n📝 DESCRIPTION ({final_state['description_word_count']} words):")
    print(f"  {final_state['description']}")
    if final_state["description_keywords"]:
        print(f"\n  Keywords used: {', '.join(final_state['description_keywords'])}")

    # Bullets
    print(f"\n🔹 BULLET POINTS:")
    for i, bullet in enumerate(final_state["bullets"], 1):
        print(f"  {i}. {bullet}")
    if final_state["bullets_issues"]:
        print("\n  Issues:")
        for issue in final_state["bullets_issues"]:
            print(f"  ⚠ {issue}")

    # Cleaning changes
    if final_state["cleaning_changes"]:
        print(f"\n🔧 CLEANING CHANGES ({len(final_state['cleaning_changes'])}):")
        for c in final_state["cleaning_changes"]:
            print(f"  - {c}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()