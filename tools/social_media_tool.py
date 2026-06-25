"""
tools/social_media_tool.py
============================
LangChain @tool — repurposes product listing into social media captions and posts.

Input  : cleaned product attributes
Output : {
    "instagram" : { "caption": str, "hashtags": list, "char_count": int }
    "facebook"  : { "post": str, "char_count": int }
    "twitter"   : { "tweet": str, "char_count": int, "valid": bool }
    "linkedin"  : { "post": str, "char_count": int }
}
"""

import json
import os
import re

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

TWITTER_LIMIT   = 280
INSTAGRAM_LIMIT = 2200

PROMPT = """You are a social media copywriter specializing in e-commerce product promotion.

Product attributes:
{attributes}

Generate social media content for this product for all 4 platforms below.
Tailor the tone to each platform's audience and style.

Platform tone guide:
- Instagram : visual, lifestyle-oriented, aspirational, emoji-friendly,
              end with 10-15 relevant hashtags on a new line
- Facebook  : conversational, slightly longer, community feel,
              can ask a question to drive engagement
- Twitter/X : punchy, concise, max {twitter_limit} characters including spaces,
              1-2 relevant hashtags inline
- LinkedIn  : professional, value-focused, speak to quality and practicality,
              no emojis, no hashtag spam

Rules for all platforms:
- Do not invent specs not in the attributes
- No hype words: amazing, best, #1, perfect
- Make it feel natural, not like a generic ad

Return ONLY this JSON, no preamble, no markdown fences:
{{
  "instagram": {{
    "caption"  : "caption text without hashtags",
    "hashtags" : ["hashtag1", "hashtag2"]
  }},
  "facebook": {{
    "post": "facebook post text"
  }},
  "twitter": {{
    "tweet": "tweet text under {twitter_limit} chars"
  }},
  "linkedin": {{
    "post": "linkedin post text"
  }}
}}"""


def _get_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "Social Media Repurposer Agent",
        },
    )


@tool
def repurpose_for_social_tool(
    attributes: dict,
    model: str = "anthropic/claude-sonnet-4",
) -> dict:
    """
    Repurposes product listing content into platform-specific social media posts.

    Args:
        attributes : cleaned product attribute dict
        model      : OpenRouter model id

    Returns dict with platform-specific posts for instagram, facebook, twitter, linkedin
    """
    prompt = PROMPT.format(
        attributes=json.dumps(attributes, indent=2),
        twitter_limit=TWITTER_LIMIT,
    )

    llm = _get_llm(model)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = re.sub(r"```json|```", "", response.content).strip()

    try:
        result = json.loads(raw)

        ig      = result.get("instagram", {})
        fb      = result.get("facebook", {})
        tw      = result.get("twitter", {})
        li      = result.get("linkedin", {})

        ig_caption   = ig.get("caption", "")
        ig_hashtags  = ig.get("hashtags", [])
        ig_full      = f"{ig_caption}\n\n{' '.join(ig_hashtags)}"

        fb_post  = fb.get("post", "")
        tw_tweet = tw.get("tweet", "")
        li_post  = li.get("post", "")

        return {
            "instagram": {
                "caption":     ig_caption,
                "hashtags":    ig_hashtags,
                "full_post":   ig_full,
                "char_count":  len(ig_full),
                "valid":       len(ig_full) <= INSTAGRAM_LIMIT,
            },
            "facebook": {
                "post":       fb_post,
                "char_count": len(fb_post),
            },
            "twitter": {
                "tweet":      tw_tweet,
                "char_count": len(tw_tweet),
                "valid":      len(tw_tweet) <= TWITTER_LIMIT,
            },
            "linkedin": {
                "post":       li_post,
                "char_count": len(li_post),
            },
        }

    except json.JSONDecodeError:
        empty = {"post": "", "char_count": 0}
        return {
            "instagram": {"caption": "", "hashtags": [], "full_post": "", "char_count": 0, "valid": True},
            "facebook":  empty,
            "twitter":   {**empty, "tweet": "", "valid": True},
            "linkedin":  empty,
        }
