# Agent Eval Dashboard

A benchmarking and evaluation dashboard for comparing LLM agents (OpenAI, Anthropic, Ollama) on the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA).

## Project Structure

- `runner.py` — runs an agent against GAIA Level 1 validation questions and saves results to `results/`
- `dashboard.py` — Streamlit dashboard that visualises accuracy, latency, tool usage, and token costs across models
- `agents/gaia_agent.py` — LangGraph-based agent with tools for web search, webpage fetching, audio transcription, image analysis, Excel reading, YouTube transcripts, and Python code execution
- `results/` — saved JSON result files from previous runs

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd ai_agent_eval_dashboard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
OPENAI_API_KEY=<your-openai-key>
ANTHROPIC_API_KEY=<your-anthropic-key>
XAI_API_KEY=<your-xai-key>
```

### 3. (Optional) Set up Ollama for local models

Ollama runs models locally — no API key needed.

1. Install Ollama: https://ollama.com/download
2. Pull the model used by this project:
   ```bash
   ollama pull llama3.1
   ```
3. Make sure Ollama is running before using `--model ollama`:
   ```bash
   ollama serve
   ```

## Running the Benchmark

```bash
python runner.py --help
```

| Flag | Description |
|------|-------------|
| `--model` | **Required.** Model to use: `openai`, `anthropic`, `ollama`, or `grok` |
| `--limit N` | Only run the first N questions (great for quick testing) |
| `--task-id ID` | Run a single question by its GAIA task ID |
| `--output-dir DIR` | Directory to save results (default: `results/`) |
| `--delay SECONDS` | Wait time between questions to avoid rate limits (default: `3`) |

### Examples

```bash
# Run all questions with OpenAI GPT-4o
python runner.py --model openai

# Quick test with first 5 questions using Anthropic Claude
python runner.py --model anthropic --limit 5

# Test with Grok
python runner.py --model grok --limit 5

# Test a single question by task ID using Ollama
python runner.py --model ollama --task-id abc123

# Save results to a custom folder with a longer delay
python runner.py --model openai --limit 10 --output-dir my_results --delay 5
```

## Viewing Results

Launch the Streamlit dashboard to compare models visually:

```bash
streamlit run dashboard.py
```

The dashboard shows accuracy, latency, tool usage, and token costs across all saved result files in `results/`.

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — model integrations (OpenAI, Anthropic, Ollama, Grok)
- [Streamlit](https://streamlit.io) — dashboard UI
- [GAIA Benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) — evaluation dataset
