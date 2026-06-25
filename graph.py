"""
graph.py — LangGraph pipeline (12 nodes)

    START → parse_input → clean_data → [flag_review] → generate_title
         → write_description → generate_bullets → generate_keywords
         → generate_faq → grammar_qa → meta_seo
         → brand_voice → ad_copy → social_media → END
"""

from langgraph.graph import StateGraph, END
from state import ProductState
from tools.input_parser_tool import parse_input_tool
from tools.data_cleaner_tool import clean_data_tool
from tools.title_generator_tool import generate_title_tool
from tools.description_writer_tool import write_description_tool
from tools.bullet_generator_tool import generate_bullets_tool
from tools.keyword_generator_tool import generate_keywords_tool
from tools.faq_generator_tool import generate_faq_tool
from tools.grammar_qa_tool import grammar_qa_tool
from tools.meta_seo_tool import generate_meta_seo_tool
from tools.brand_voice_tool import check_brand_voice_tool
from tools.ad_copy_tool import generate_ad_copy_tool
from tools.social_media_tool import repurpose_for_social_tool


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_input_node(state: ProductState) -> dict:
    print("\n[ 1/12] Parsing input...")
    result = parse_input_tool.invoke({"raw_input": state["raw_input"], "model": state["model"]})
    print(f"        ✓ Extracted {len(result)} attributes")
    return {"parsed_attributes": result}


def clean_data_node(state: ProductState) -> dict:
    print("\n[ 2/12] Cleaning data...")
    result = clean_data_tool.invoke({"attributes": state["parsed_attributes"], "model": state["model"]})
    if result["changes"]:
        print(f"        ✓ Applied {len(result['changes'])} fix(es)")
    if result["flags"]:
        print(f"        ⚠ {len(result['flags'])} flag(s) need review")
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
    print(f"\n[ 3/12] Generating {state['title_count']} title(s)...")
    results = generate_title_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "count": state["title_count"],
        "model": state["model"],
    })
    ok = sum(1 for r in results if r["valid"])
    print(f"        ✓ {len(results)} title(s) — {ok} valid")
    return {"titles": results}


def write_description_node(state: ProductState) -> dict:
    print(f"\n[ 4/12] Writing description...")
    result = write_description_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "output_format": state["output_format"],
        "model": state["model"],
    })
    print(f"        ✓ {result['word_count']} words")
    return {
        "description":            result["description"],
        "description_word_count": result["word_count"],
        "description_keywords":   result["keywords_used"],
    }


def generate_bullets_node(state: ProductState) -> dict:
    print(f"\n[ 5/12] Generating {state['bullet_count']} bullets...")
    result = generate_bullets_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "count": state["bullet_count"],
        "model": state["model"],
    })
    status = "✓ all valid" if result["valid"] else f"⚠ {len(result['issues'])} issue(s)"
    print(f"        {status}")
    return {
        "bullets":        result["bullets"],
        "bullets_valid":  result["valid"],
        "bullets_issues": result["issues"],
    }


def generate_keywords_node(state: ProductState) -> dict:
    print(f"\n[ 6/12] Generating keywords & tags...")
    result = generate_keywords_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "model": state["model"],
    })
    total = len(result["primary_keywords"]) + len(result["secondary_keywords"])
    print(f"        ✓ {total} keywords, {len(result['tags'])} tags")
    return {
        "primary_keywords":   result["primary_keywords"],
        "secondary_keywords": result["secondary_keywords"],
        "tags":               result["tags"],
        "backend_keywords":   result["backend_keywords"],
        "keywords_valid":     result["valid"],
    }


def generate_faq_node(state: ProductState) -> dict:
    print(f"\n[ 7/12] Generating {state['faq_count']} FAQs...")
    result = generate_faq_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "marketplace": state["marketplace"],
        "count": state["faq_count"],
        "model": state["model"],
    })
    print(f"        ✓ {result['faq_count']} FAQs — topics: {', '.join(result['categories'])}")
    return {
        "faqs":           result["faqs"],
        "faq_categories": result["categories"],
    }


def grammar_qa_node(state: ProductState) -> dict:
    print(f"\n[ 8/12] Running grammar & spelling QA...")
    result = grammar_qa_tool.invoke({
        "text": state["description"],
        "content_type": "product description",
        "model": state["model"],
    })
    status = "✓ clean" if result["clean"] else f"⚠ {result['issue_count']} issue(s) fixed"
    print(f"        {status} — readability: {result['readability']}")
    return {
        "corrected_description": result["corrected_text"],
        "grammar_issues":        result["issues_found"],
        "grammar_clean":         result["clean"],
        "readability":           result["readability"],
    }


def meta_seo_node(state: ProductState) -> dict:
    print(f"\n[ 9/12] Generating meta SEO tags...")
    existing_title = state["titles"][0]["title"] if state["titles"] else ""
    result = generate_meta_seo_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "existing_title": existing_title,
        "model": state["model"],
    })
    t = "✓" if result["title_valid"] else "⚠"
    d = "✓" if result["desc_valid"]  else "⚠"
    print(f"        {t} meta title ({result['meta_title_len']} chars)  "
          f"{d} meta desc ({result['meta_desc_len']} chars)")
    return {
        "meta_title":       result["meta_title"],
        "meta_description": result["meta_description"],
        "meta_title_valid": result["title_valid"],
        "meta_desc_valid":  result["desc_valid"],
        "meta_issues":      result["issues"],
    }


def brand_voice_node(state: ProductState) -> dict:
    print(f"\n[10/12] Checking brand voice...")
    desc = state.get("corrected_description") or state["description"]
    result = check_brand_voice_tool.invoke({
        "content":      desc,
        "brand_guide":  state["brand_guide"],
        "content_type": "product description",
        "model":        state["model"],
    })
    status = "✓ on-brand" if result["on_brand"] else f"⚠ off-brand (score: {result['score']}/100)"
    print(f"        {status}")
    return {
        "brand_score":       result["score"],
        "brand_on_brand":    result["on_brand"],
        "brand_issues":      result["issues"],
        "brand_rewritten":   result["rewritten"],
        "brand_suggestions": result["suggestions"],
    }


def ad_copy_node(state: ProductState) -> dict:
    print(f"\n[11/12] Generating ad copy ({state['ad_platform']})...")
    result = generate_ad_copy_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "platform":   state["ad_platform"],
        "audience":   state["ad_audience"],
        "model":      state["model"],
    })
    print(f"        ✓ {result['headline_count']} headlines + short & long copy")
    return {
        "ad_headlines":  result["headlines"],
        "ad_short_copy": result["short_copy"],
        "ad_long_copy":  result["long_copy"],
        "ad_cta":        result["call_to_action"],
    }


def social_media_node(state: ProductState) -> dict:
    print(f"\n[12/12] Repurposing for social media...")
    result = repurpose_for_social_tool.invoke({
        "attributes": state["cleaned_attributes"],
        "model":      state["model"],
    })
    tw_valid = "✓" if result["twitter"]["valid"] else "⚠ over limit"
    print(f"        ✓ Instagram · Facebook · LinkedIn · Twitter {tw_valid}")
    return {
        "social_instagram": result["instagram"],
        "social_facebook":  result["facebook"],
        "social_twitter":   result["twitter"],
        "social_linkedin":  result["linkedin"],
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_flag(state: ProductState) -> str:
    return "flag" if state.get("cleaning_flags") else "title"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(ProductState)

    graph.add_node("parse_input",       parse_input_node)
    graph.add_node("clean_data",        clean_data_node)
    graph.add_node("flag_review",       flag_review_node)
    graph.add_node("generate_title",    generate_title_node)
    graph.add_node("write_description", write_description_node)
    graph.add_node("generate_bullets",  generate_bullets_node)
    graph.add_node("generate_keywords", generate_keywords_node)
    graph.add_node("generate_faq",      generate_faq_node)
    graph.add_node("grammar_qa",        grammar_qa_node)
    graph.add_node("meta_seo",          meta_seo_node)
    graph.add_node("brand_voice",       brand_voice_node)
    graph.add_node("ad_copy",           ad_copy_node)
    graph.add_node("social_media",      social_media_node)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input",       "clean_data")
    graph.add_conditional_edges(
        "clean_data",
        should_flag,
        {"flag": "flag_review", "title": "generate_title"},
    )
    graph.add_edge("flag_review",       "generate_title")
    graph.add_edge("generate_title",    "write_description")
    graph.add_edge("write_description", "generate_bullets")
    graph.add_edge("generate_bullets",  "generate_keywords")
    graph.add_edge("generate_keywords", "generate_faq")
    graph.add_edge("generate_faq",      "grammar_qa")
    graph.add_edge("grammar_qa",        "meta_seo")
    graph.add_edge("meta_seo",          "brand_voice")
    graph.add_edge("brand_voice",       "ad_copy")
    graph.add_edge("ad_copy",           "social_media")
    graph.add_edge("social_media",      END)

    return graph.compile()


pipeline = build_graph()