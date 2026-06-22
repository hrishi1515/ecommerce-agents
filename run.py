"""
run.py — Entry point for the full LangGraph agent pipeline
===========================================================
Usage:
    python3 run.py --input product.json --marketplace amazon
    python3 run.py --input catalog.csv  --marketplace walmart --count 3
    python3 run.py --input product.txt  --model openai/gpt-4o-mini
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
    parser = argparse.ArgumentParser(description="Run the full agent pipeline (parse → clean → title)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to .json, .csv, or text file")
    group.add_argument("--text",  help='Raw product description string')
    group.add_argument("--stdin", action="store_true", help="Read from stdin (pipe input)")

    parser.add_argument("--marketplace", default="generic",
                        choices=["amazon", "walmart", "etsy", "generic"])
    parser.add_argument("--count", type=int, default=1, help="Number of title options")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    # Resolve raw_input
    if args.stdin:
        raw_input = sys.stdin.read().strip()
    elif args.text:
        raw_input = args.text
    else:
        raw_input = args.input   # file path — input_parser_tool handles loading

    # Build initial state
    initial_state = {
        "raw_input":          raw_input,
        "parsed_attributes":  {},
        "cleaned_attributes": {},
        "cleaning_changes":   [],
        "cleaning_flags":     [],
        "titles":             [],
        "marketplace":        args.marketplace,
        "title_count":        args.count,
        "model":              args.model,
        "errors":             [],
    }

    print("=" * 55)
    print("  Agent Pipeline — LangGraph")
    print("=" * 55)

    # Run the graph
    final_state = pipeline.invoke(initial_state)

    # --- Print results ---
    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)

    print(f"\nCleaned attributes ({len(final_state['cleaned_attributes'])} fields):")
    print(json.dumps(final_state["cleaned_attributes"], indent=2))

    if final_state["cleaning_changes"]:
        print(f"\nCleaning changes ({len(final_state['cleaning_changes'])}):")
        for c in final_state["cleaning_changes"]:
            print(f"  - {c}")

    print(f"\nGenerated titles for '{args.marketplace}':")
    for i, r in enumerate(final_state["titles"], 1):
        status = "✓ OK" if r["valid"] else "⚠ REVIEW"
        print(f"\n  {i}. [{status}] ({r['length']} chars)")
        print(f"     {r['title']}")
        for issue in r.get("issues", []):
            print(f"     ⚠ {issue}")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
