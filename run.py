"""
run.py — Entry point for the full LangGraph agent pipeline (12 agents)
=======================================================================
Usage:
    python3 run.py --input product.json --marketplace amazon
    python3 run.py --input product.json --marketplace amazon --count 2 --bullets 5 --faqs 5
    python3 run.py --text "Nike black running shoes, mesh upper, size 10"
    python3 run.py --input product.json --ad-platform facebook --ad-audience "fitness enthusiasts"

Requires:
    export OPENROUTER_API_KEY=your_key_here
"""

import argparse
import json
import os
import sys

from graph import pipeline

DEFAULT_BRAND_GUIDE = (
    "Professional yet approachable tone. Clear and benefit-focused. "
    "No slang, no hype words, no exclamation marks. "
    "Sentences should be concise and easy to understand."
)


def main():
    parser = argparse.ArgumentParser(description="Run the full agent pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  help="Path to .json, .csv, or text file")
    group.add_argument("--text",   help="Raw product description string")
    group.add_argument("--stdin",  action="store_true", help="Read from stdin")

    parser.add_argument("--marketplace",  default="generic",
                        choices=["amazon", "walmart", "etsy", "generic"])
    parser.add_argument("--count",        type=int, default=1,     help="Number of title options")
    parser.add_argument("--bullets",      type=int, default=5,     help="Number of bullet points")
    parser.add_argument("--faqs",         type=int, default=5,     help="Number of FAQ pairs")
    parser.add_argument("--format",       default="plain",         choices=["plain", "html"])
    parser.add_argument("--ad-platform",  default="generic",
                        choices=["amazon", "google", "facebook", "generic"])
    parser.add_argument("--ad-audience",  default="general shoppers",
                        help='Target audience e.g. "fitness enthusiasts aged 25-40"')
    parser.add_argument("--brand-guide",  default=DEFAULT_BRAND_GUIDE,
                        help="Brand style guide as a text string")
    parser.add_argument("--model",        default="anthropic/claude-sonnet-4")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    raw_input = sys.stdin.read().strip() if args.stdin else (args.text or args.input)

    initial_state = {
        "raw_input":             raw_input,
        "parsed_attributes":     {},
        "cleaned_attributes":    {},
        "cleaning_changes":      [],
        "cleaning_flags":        [],
        "titles":                [],
        "description":           "",
        "description_word_count": 0,
        "description_keywords":  [],
        "bullets":               [],
        "bullets_valid":         True,
        "bullets_issues":        [],
        "primary_keywords":      [],
        "secondary_keywords":    [],
        "tags":                  [],
        "backend_keywords":      "",
        "keywords_valid":        True,
        "faqs":                  [],
        "faq_categories":        [],
        "grammar_issues":        [],
        "grammar_clean":         True,
        "corrected_description": "",
        "readability":           "easy",
        "meta_title":            "",
        "meta_description":      "",
        "meta_title_valid":      True,
        "meta_desc_valid":       True,
        "meta_issues":           [],
        "brand_score":           100,
        "brand_on_brand":        True,
        "brand_issues":          [],
        "brand_rewritten":       "",
        "brand_suggestions":     [],
        "ad_headlines":          [],
        "ad_short_copy":         "",
        "ad_long_copy":          "",
        "ad_cta":                "",
        "social_instagram":      {},
        "social_facebook":       {},
        "social_twitter":        {},
        "social_linkedin":       {},
        "marketplace":           args.marketplace,
        "title_count":           args.count,
        "bullet_count":          args.bullets,
        "faq_count":             args.faqs,
        "output_format":         args.format,
        "ad_platform":           args.ad_platform,
        "ad_audience":           args.ad_audience,
        "brand_guide":           args.brand_guide,
        "model":                 args.model,
        "errors":                [],
    }

    print("=" * 60)
    print("  Agent Pipeline — LangGraph (12 Agents)")
    print("=" * 60)

    final_state = pipeline.invoke(initial_state)

    # ── Results ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # Titles
    print(f"\n📌 TITLES ({args.marketplace.upper()}):")
    for i, r in enumerate(final_state["titles"], 1):
        status = "✓" if r["valid"] else "⚠"
        print(f"  {i}. {status} ({r['length']} chars) {r['title']}")

    # Description
    desc = final_state.get("corrected_description") or final_state["description"]
    wc   = final_state["description_word_count"]
    note = "" if final_state["grammar_clean"] else " — grammar corrected"
    print(f"\n📝 DESCRIPTION ({wc} words{note} | readability: {final_state['readability']}):")
    print(f"  {desc}")

    # Brand voice
    score = final_state["brand_score"]
    bv_status = "✓ on-brand" if final_state["brand_on_brand"] else f"⚠ score {score}/100"
    print(f"\n🎨 BRAND VOICE: {bv_status}")
    if not final_state["brand_on_brand"]:
        print(f"  Rewritten: {final_state['brand_rewritten']}")
        for s in final_state["brand_suggestions"]:
            print(f"  💡 {s}")

    # Bullets
    print(f"\n🔹 BULLET POINTS:")
    for i, bullet in enumerate(final_state["bullets"], 1):
        print(f"  {i}. {bullet}")

    # Keywords
    print(f"\n🔑 KEYWORDS:")
    print(f"  Primary   : {', '.join(final_state['primary_keywords'])}")
    print(f"  Secondary : {', '.join(final_state['secondary_keywords'])}")
    print(f"  Tags      : {', '.join(final_state['tags'])}")
    if final_state["backend_keywords"]:
        valid_str = "✓" if final_state["keywords_valid"] else "⚠ over limit"
        print(f"  Backend   : {final_state['backend_keywords']} ({valid_str})")

    # FAQs
    print(f"\n❓ FAQs (topics: {', '.join(final_state['faq_categories'])}):")
    for i, faq in enumerate(final_state["faqs"], 1):
        print(f"\n  Q{i}: {faq.get('question', '')}")
        print(f"  A{i}: {faq.get('answer', '')}")

    # Meta SEO
    t_s = "✓" if final_state["meta_title_valid"] else "⚠"
    d_s = "✓" if final_state["meta_desc_valid"]  else "⚠"
    print(f"\n🔍 META SEO:")
    print(f"  {t_s} Title ({len(final_state['meta_title'])} chars): {final_state['meta_title']}")
    print(f"  {d_s} Desc  ({len(final_state['meta_description'])} chars): {final_state['meta_description']}")

    # Ad Copy
    print(f"\n📣 AD COPY ({args.ad_platform.upper()}):")
    print(f"  Headlines:")
    for h in final_state["ad_headlines"]:
        print(f"    • {h}")
    print(f"  Short : {final_state['ad_short_copy']}")
    print(f"  Long  : {final_state['ad_long_copy']}")
    print(f"  CTA   : {final_state['ad_cta']}")

    # Social Media
    print(f"\n📱 SOCIAL MEDIA:")
    ig = final_state["social_instagram"]
    fb = final_state["social_facebook"]
    tw = final_state["social_twitter"]
    li = final_state["social_linkedin"]
    tw_valid = "✓" if tw.get("valid") else f"⚠ {tw.get('char_count', 0)}/280 chars"

    print(f"\n  Instagram ({ig.get('char_count', 0)} chars):")
    print(f"    {ig.get('caption', '')}")
    print(f"    {' '.join(ig.get('hashtags', []))}")

    print(f"\n  Facebook ({fb.get('char_count', 0)} chars):")
    print(f"    {fb.get('post', '')}")

    print(f"\n  Twitter/X ({tw.get('char_count', 0)} chars — {tw_valid}):")
    print(f"    {tw.get('tweet', '')}")

    print(f"\n  LinkedIn ({li.get('char_count', 0)} chars):")
    print(f"    {li.get('post', '')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()