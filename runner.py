import argparse
import json
import logging
import os
import time
from datetime import datetime

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Run the GAIA benchmark agent and evaluate performance across models.",
    epilog="""
examples:
  # Run all questions with OpenAI GPT-4o
  python runner.py --model openai

  # Quick test with first 5 questions using Anthropic Claude
  python runner.py --model anthropic --limit 5

  # Test a single question by task ID using Ollama (requires Ollama running locally)
  python runner.py --model ollama --task-id abc123

  # Run with Grok
  python runner.py --model grok --limit 5

  # Save results to a custom directory with a longer delay between questions
  python runner.py --model openai --limit 10 --output-dir my_results --delay 5

required env vars:
  OPENAI_API_KEY    — needed for --model openai (and audio/image tools for all models)
  ANTHROPIC_API_KEY — needed for --model anthropic
  XAI_API_KEY       — needed for --model grok, get yours at https://console.x.ai
  OLLAMA running locally at http://localhost:11434 — needed for --model ollama
    install: https://ollama.com/download
    pull model: ollama pull llama3.1
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--model",
    choices=["openai", "anthropic", "ollama", "grok"],
    required=True,
    help="model provider to use: openai (gpt-4o), anthropic (claude-sonnet-5), ollama (llama3.1 local), or grok (grok-3)",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    metavar="N",
    help="only run the first N questions (useful for quick testing)",
)
parser.add_argument(
    "--task-id",
    default=None,
    metavar="TASK_ID",
    help="run a single question by its GAIA task ID (overrides --limit)",
)
parser.add_argument(
    "--output-dir",
    default="results",
    metavar="DIR",
    help="directory to save result JSON files (default: results/)",
)
parser.add_argument(
    "--delay",
    type=float,
    default=3.0,
    metavar="SECONDS",
    help="seconds to wait between questions to avoid rate limits (default: 3)",
)
args = parser.parse_args()



os.environ["MODEL_PROVIDER"] = args.model  # must be set before importing the agent

from agents.gaia_agent import run

API_BASE = "https://agents-course-unit4-scoring.hf.space"
GAIA_REPO = "gaia-benchmark/GAIA"

ds = load_dataset(GAIA_REPO, "2023_level1", split="validation")
FILE_PATHS = {r["task_id"]: r["file_path"] for r in ds if r.get("file_path")}
EXPECTED = {r["task_id"]: r["Final answer"] for r in ds}


def get_questions():
    response = requests.get(f"{API_BASE}/questions")
    response.raise_for_status()
    return response.json()


def attachment_path(task_id: str):
    rel = FILE_PATHS.get(task_id)
    if not rel:
        return None
    try:
        return hf_hub_download(repo_id=GAIA_REPO, filename=rel, repo_type="dataset")
    except Exception as e:
        logger.warning("Could not download attachment for %s: %s", task_id, type(e).__name__)
        return None


if __name__ == "__main__":
    questions = get_questions()

    if args.task_id:
        questions = [q for q in questions if q["task_id"] == args.task_id]
        if not questions:
            print(f"No question found with task_id '{args.task_id}'")
            exit(1)
    elif args.limit:
        questions = questions[:args.limit]

    print(f"Running {len(questions)} question(s) with model: {args.model}")

    records = []
    for i, q in enumerate(questions, 1):
        task_id = q["task_id"]
        file_path = attachment_path(task_id) if q.get("file_name") else None
        expected = str(EXPECTED.get(task_id, ""))

        print(f"[{i}/{len(questions)}] {task_id}")
        try:
            out = run(q["question"], file_path)
            error = None
        except Exception as e:
            out = {"answer": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0, "latency_s": 0}
            error = type(e).__name__

        correct = out["answer"].strip().lower() == expected.strip().lower()
        records.append({
            "task_id": task_id,
            "question": q["question"],
            "expected": expected,
            "got": out["answer"],
            "correct": correct,
            "latency_s": out["latency_s"],
            "tool_calls": out["tool_calls"],
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "error": error,
        })
        print(f"  {'✅' if correct else '❌'}  got: {out['answer']}  expected: {expected}")
        if i < len(questions):
            time.sleep(args.delay)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"{args.model}_{stamp}.json")
    with open(path, "w") as f:
        json.dump({"model": args.model, "timestamp": stamp, "records": records}, f, indent=2)

    passed = sum(r["correct"] for r in records)
    print(f"\n{passed}/{len(records)} correct. Results saved to {path}")