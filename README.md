# Agent Eval Dashboard

A benchmarking and evaluation dashboard for comparing LLM agents (OpenAI, Anthropic, Ollama) on the [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA).

## Overview

- **runner.py** — runs an agent against GAIA Level 1 validation questions and saves results to `results/`
- **dashboard.py** — Streamlit dashboard that visualises accuracy, latency, tool usage, and token costs across models
- **agents/gaia_agent.py** — LangGraph-based agent with tools for web search, webpage fetching, audio transcription, image analysis, Excel reading, YouTube transcripts, and Python code execution

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your API keys to .env
```

## Usage

Run the agent against the benchmark:
```bash
python runner.py --model openai       # GPT-4o
python runner.py --model anthropic    # Claude Sonnet
python runner.py --model ollama       # Llama 3.1 (local)
python runner.py --model openai --limit 10  # first 10 questions only
```

Launch the dashboard:
```bash
streamlit run dashboard.py
```

## Environment Variables

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [Streamlit](https://streamlit.io) — dashboard UI
- [GAIA Benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) — evaluation dataset
