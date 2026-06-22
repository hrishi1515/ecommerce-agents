LangGraph pipeline: Input Parser → Data Cleaner → Title Generator


# How to use it 

git clone https://github.com/hrishi1515/ecommerce-agents.git
cd ecommerce-agents
conda create -n title-gen python=3.11 -y
conda activate title-gen
pip install langchain langgraph langchain-openai langchain-core requests
export OPENROUTER_API_KEY=their_key_here
python3 run.py --input sample_product.json --marketplace amazon
