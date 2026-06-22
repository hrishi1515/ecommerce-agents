LangGraph pipeline: Input Parser → Data Cleaner → Title Generator


## Setup

```bash
git clone https://github.com/hrishi1515/ecommerce-agents.git
cd ecommerce-agents
conda create -n title-gen python=3.11 -y
conda activate title-gen
pip install langchain langgraph langchain-openai langchain-core requests
export OPENROUTER_API_KEY=your_key_here
```

## Run

```bash
# From a JSON file
python3 run.py --input sample_product.json --marketplace amazon

# Multiple title options
python3 run.py --input sample_product.json --marketplace amazon --count 3

# From a text file
python3 run.py --input product.txt --marketplace walmart

# From raw text
python3 run.py --text "Nike black shoes size 10" --marketplace etsy

# From stdin
echo "32oz stainless bottle vacuum insulated" | python3 run.py --stdin
```

