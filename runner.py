import argparse
import json
import os
import time
from datetime import datetime

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["openai", "anthropic", "ollama"], required=True)
parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
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
        print(f"Could not download attachment for {task_id}: {e}")
        return None


if __name__ == "__main__":
    questions = get_questions()
    if args.limit:
        questions = questions[:args.limit]

    records = []
    for q in questions:
        task_id = q["task_id"]
        file_path = attachment_path(task_id) if q.get("file_name") else None
        expected = str(EXPECTED.get(task_id, ""))

        try:
            out = run(q["question"], file_path)
            error = None
        except Exception as e:
            out = {"answer": "", "tool_calls": [], "input_tokens": 0, "output_tokens": 0, "latency_s": 0}
            error = str(e)

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
        print(f"{'✅' if correct else '❌'} {task_id}  got: {out['answer']}  expected: {expected}")
        time.sleep(3)

    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{args.model}_{stamp}.json"
    with open(path, "w") as f:
        json.dump({"model": args.model, "timestamp": stamp, "records": records}, f, indent=2)

    passed = sum(r["correct"] for r in records)
    print(f"\n{passed}/{len(records)} correct. Saved to {path}")