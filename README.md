# E-Commerce Agent Pipeline

A fully automated product content pipeline built with **LangChain** and **LangGraph**.  
Takes raw product data as input and generates everything needed for a complete marketplace listing.

---

## Agents (21 total — README covers first 15)

| # | Agent | What it does |
|---|-------|-------------|
| 1 | Input Parser | Accepts JSON, CSV, plain text, or raw string |
| 2 | Data Cleaner | Normalizes units, fixes casing, dedupes, flags issues |
| 3 | Title Generator | Marketplace-optimized titles (Amazon, Walmart, Etsy) |
| 4 | Description Writer | SEO-friendly product descriptions |
| 5 | Bullet Generator | Scannable feature bullet points |
| 6 | Keyword Generator | Primary, secondary keywords + backend search tags |
| 7 | FAQ Generator | Auto product Q&A block |
| 8 | Grammar QA | Spelling, grammar, readability check |
| 9 | Meta SEO | Meta title + meta description for search snippets |
| 10 | Brand Voice Checker | Checks and rewrites content to match brand style |
| 11 | Ad Copy Generator | Sponsored ad headlines and copy |
| 12 | Social Media Repurposer | Instagram, Facebook, Twitter, LinkedIn posts |
| 13 | Review Response | Drafts replies to customer reviews |
| 14 | Customer Q&A Responder | Answers buyer questions from product data |
| 15 | Email Generator | Marketing email copy (launch, promo, newsletter, etc.) |

---

## Setup

```bash
git clone https://github.com/hrishi1515/ecommerce-agents.git
cd ecommerce-agents
conda create -n title-gen python=3.11 -y
conda activate title-gen
pip install langchain langgraph langchain-openai langchain-core requests
export OPENROUTER_API_KEY=your_key_here
```

---

## Run

```bash
# Basic run
python3 run.py --input sample_product.json --marketplace amazon

# Multiple title options + custom bullet/FAQ count
python3 run.py --input sample_product.json --marketplace amazon --count 3 --bullets 5 --faqs 5

# From raw text (no file needed)
python3 run.py --text "Nike black running shoes, mesh upper, size 10"

# From stdin
echo "32oz stainless bottle, vacuum insulated" | python3 run.py --stdin

# With customer review to respond to
python3 run.py --input sample_product.json \
  --review "Bottle leaks after 2 weeks, very disappointed" \
  --review-rating 2

# With buyer question to answer
python3 run.py --input sample_product.json \
  --question "Does this fit in a standard car cup holder?"

# With specific email type
python3 run.py --input sample_product.json --email-type promo

# With custom ad platform and audience
python3 run.py --input sample_product.json \
  --ad-platform facebook \
  --ad-audience "fitness enthusiasts aged 25-40"

# With custom brand guide
python3 run.py --input sample_product.json \
  --brand-guide "Bold and confident. Short punchy sentences. Use active verbs."

# Cheaper/faster model
python3 run.py --input sample_product.json --model openai/gpt-4o-mini
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--marketplace` | `generic` | `amazon` / `walmart` / `etsy` / `generic` |
| `--count` | `1` | Number of title options to generate |
| `--bullets` | `5` | Number of bullet points |
| `--faqs` | `5` | Number of FAQ pairs |
| `--format` | `plain` | `plain` / `html` for description output |
| `--ad-platform` | `generic` | `amazon` / `google` / `facebook` / `generic` |
| `--ad-audience` | `general shoppers` | Target audience for ad copy |
| `--brand-guide` | built-in default | Your brand style guide as a plain text string |
| `--review` | _(empty)_ | Customer review text to respond to |
| `--review-rating` | `5` | Star rating 1–5 |
| `--question` | _(empty)_ | Buyer question to answer from product data |
| `--email-type` | `launch` | `launch` / `promo` / `newsletter` / `restock` / `abandoned` |
| `--model` | `anthropic/claude-sonnet-4` | Any OpenRouter model id |

---

## Agent Details

### 1. Input Parser
Accepts any input format and converts it to structured product attributes.
- JSON file, CSV file, plain text file, raw string, or stdin
- For unstructured text, uses LLM to extract attributes automatically
- No API call needed for JSON/CSV inputs

### 2. Data Cleaner
Two-layer cleaning — rule-based first, then LLM for fuzzy fixes.
- Rule-based: trims whitespace, normalizes units (oz/lb/in/cm), Title Cases fields, dedupes lists, drops empty fields
- LLM-based: fixes typos, standardizes inconsistent vocab (e.g. "SS" → "Stainless Steel")
- Flags suspicious or contradictory data for human review instead of silently changing it
- Outputs a full audit trail of every change made

### 3. Title Generator
Generates keyword-optimized, marketplace-compliant product titles.
- Amazon: max 200 chars, Brand + Type + Features + Size/Color
- Walmart: max 100 chars, simplified format
- Etsy: max 140 chars, keyword-rich with lifestyle tone
- Validates each title: checks length, banned words, ALL CAPS

### 4. Description Writer
Generates persuasive, SEO-optimized product descriptions.
- Leads with the strongest customer benefit (not a feature)
- Naturally weaves in keywords extracted from attributes
- Supports plain text or HTML output
- Marketplace-aware tone: Amazon = factual, Etsy = lifestyle, Walmart = simple

### 5. Bullet Generator
Generates scannable "About this item" bullet points.
- Amazon style: ALL-CAPS lead keyword + em dash + benefit
- Validates each bullet against character limits (500 chars Amazon, 300 generic)
- Flags banned/promotional words per bullet

### 6. Keyword Generator
Generates search-optimized keywords and tags.
- Primary keywords (5–8): highest search volume terms
- Secondary keywords (8–12): long-tail and supporting terms
- Tags (10–13): short marketplace tags
- Backend keywords: space-separated string for hidden search fields
- Validates against marketplace character limits (Amazon: 249 chars)

### 7. FAQ Generator
Generates a product Q&A block focused on reducing purchase hesitation.
- Covers: compatibility, sizing, care, materials, warranty, usage
- Answers strictly from product data — never invents specs
- Returns which topics were covered so you can spot gaps

### 8. Grammar QA
Reviews generated content for spelling, grammar, and readability issues.
- Fixes: spelling mistakes, grammar errors, punctuation, awkward phrasing
- Rates readability: easy / moderate / complex
- Returns corrected version + list of specific issues fixed

### 9. Meta SEO Generator
Generates meta title and meta description for search engine snippets.
- Meta title: max 60 characters, keyword + brand
- Meta description: max 160 characters, value proposition + soft CTA
- Controls exactly how your product appears in Google search results
- Validates both against character limits

### 10. Brand Voice Checker
Checks content against your brand style guide and rewrites if off-brand.
- Scores brand alignment 0–100
- Flags specific off-brand words, tone issues, and style violations
- Rewrites content to be fully on-brand
- Gives actionable suggestions for future content
- Customize via `--brand-guide "your style guide here"`

### 11. Ad Copy Generator
Generates sponsored ad headlines and copy.
- 5 short headlines (max 30 chars each)
- Short copy (1–2 sentences) for small placements
- Long copy (3–4 sentences) for larger placements
- Call-to-action button text
- Platform-specific tone: Amazon, Google, Facebook, or generic

### 12. Social Media Repurposer
Repurposes product listing into platform-specific social posts.
- Instagram: caption + 10–15 hashtags, lifestyle tone
- Facebook: conversational, engagement-focused with a question
- Twitter/X: max 280 chars, punchy, validated
- LinkedIn: professional, value-focused, no emoji/hashtag spam

### 13. Review Response Agent
Drafts professional responses to customer reviews.
- 4–5 stars: sincere thank-you, highlights what they loved
- 1–2 stars: apology + clear resolution path
- 3 stars: acknowledges concern + shows you value feedback
- Auto-flags safety issues or legal threats for human escalation
- Never offers specific discounts or refund amounts

### 14. Customer Q&A Responder
Answers buyer questions strictly from product data.
- Returns confidence level: high / medium / low
- Flags questions it can't answer (pricing, shipping, returns) for support
- Never invents specs or makes claims not in the data

### 15. Email Generator
Generates full marketing email copy for a product.
- Subject line (max 60 chars) + preview text (max 90 chars) + body + CTA
- 5 email types: launch, promo, newsletter, restock, cart abandonment
- Each type has different tone and structure rules
- Body target: 120–180 words, short scannable paragraphs

---

## Project Structure

```
agents/
├── run.py                             # entry point
├── graph.py                           # LangGraph pipeline
├── state.py                           # shared state schema
├── sample_product.json                # sample input for testing
└── tools/
    ├── __init__.py
    ├── input_parser_tool.py           # agent 1
    ├── data_cleaner_tool.py           # agent 2
    ├── title_generator_tool.py        # agent 3
    ├── description_writer_tool.py     # agent 4
    ├── bullet_generator_tool.py       # agent 5
    ├── keyword_generator_tool.py      # agent 6
    ├── faq_generator_tool.py          # agent 7
    ├── grammar_qa_tool.py             # agent 8
    ├── meta_seo_tool.py               # agent 9
    ├── brand_voice_tool.py            # agent 10
    ├── ad_copy_tool.py                # agent 11
    ├── social_media_tool.py           # agent 12
    ├── review_response_tool.py        # agent 13
    ├── qa_responder_tool.py           # agent 14
    ├── email_generator_tool.py        # agent 15
    ├── variant_copy_tool.py           # agent 16
    ├── alt_text_tool.py               # agent 17
    ├── video_script_tool.py           # agent 18
    ├── compliance_checker_tool.py     # agent 19
    ├── seo_audit_tool.py              # agent 20
    └── review_insight_tool.py         # agent 21
```

---

## Switching Models

Any OpenRouter model works via `--model`:

```bash
# Fast and cheap
python3 run.py --input sample_product.json --model openai/gpt-4o-mini

# Most capable
python3 run.py --input sample_product.json --model anthropic/claude-sonnet-4

# Open source
python3 run.py --input sample_product.json --model meta-llama/llama-3.3-70b-instruct
```

---

## Requirements

- Python 3.11+
- OpenRouter API key — [openrouter.ai](https://openrouter.ai)
- Dependencies: `langchain langgraph langchain-openai langchain-core requests`
