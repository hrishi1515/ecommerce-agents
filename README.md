# E-Commerce Agent Pipeline

A fully automated product content pipeline built with **LangChain** and **LangGraph**.  
Takes raw product data as input and generates everything needed for a complete marketplace listing.

---

## Agents (15 total)

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

# From raw text
python3 run.py --text "Nike black running shoes, mesh upper, size 10"

# From stdin
echo "32oz stainless bottle, vacuum insulated" | python3 run.py --stdin

# With customer review response
python3 run.py --input sample_product.json \
  --review "Bottle leaks after 2 weeks, very disappointed" \
  --review-rating 2

# With buyer question
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
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--marketplace` | `generic` | `amazon` / `walmart` / `etsy` / `generic` |
| `--count` | `1` | Number of title options |
| `--bullets` | `5` | Number of bullet points |
| `--faqs` | `5` | Number of FAQ pairs |
| `--format` | `plain` | `plain` / `html` for description output |
| `--ad-platform` | `generic` | `amazon` / `google` / `facebook` / `generic` |
| `--ad-audience` | `general shoppers` | Target audience for ad copy |
| `--brand-guide` | built-in default | Your brand style guide as a string |
| `--review` | _(empty)_ | Customer review text to respond to |
| `--review-rating` | `5` | Star rating 1-5 |
| `--question` | _(empty)_ | Buyer question to answer |
| `--email-type` | `launch` | `launch` / `promo` / `newsletter` / `restock` / `abandoned` |
| `--model` | `anthropic/claude-sonnet-4` | Any OpenRouter model id |

---

## Project Structure

```
agents/
├── run.py                          # entry point
├── graph.py                        # LangGraph pipeline
├── state.py                        # shared state schema
├── sample_product.json             # sample input for testing
└── tools/
    ├── __init__.py
    ├── input_parser_tool.py
    ├── data_cleaner_tool.py
    ├── title_generator_tool.py
    ├── description_writer_tool.py
    ├── bullet_generator_tool.py
    ├── keyword_generator_tool.py
    ├── faq_generator_tool.py
    ├── grammar_qa_tool.py
    ├── meta_seo_tool.py
    ├── brand_voice_tool.py
    ├── ad_copy_tool.py
    ├── social_media_tool.py
    ├── review_response_tool.py
    ├── qa_responder_tool.py
    └── email_generator_tool.py
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
