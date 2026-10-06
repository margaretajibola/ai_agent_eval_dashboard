import glob
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agent Eval Dashboard", layout="wide")
st.title("Agent Evaluation Dashboard")

files = sorted(glob.glob("results/*.json"))
if not files:
    st.warning("No results yet. Run runner.py first.")
    st.stop()

# load every run, keep the latest run per model
runs = {}
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    runs[data["model"]] = data  # files are sorted by timestamp, so the last one wins

rows = []
for model, data in runs.items():
    for r in data["records"]:
        rows.append({"model": model, **r})
df = pd.DataFrame(rows)
df["tool_count"] = df["tool_calls"].apply(len)

# summary per model
summary = df.groupby("model").agg(
    questions=("task_id", "count"),
    correct=("correct", "sum"),
    accuracy=("correct", "mean"),
    avg_latency_s=("latency_s", "mean"),
    avg_tool_calls=("tool_count", "mean"),
    input_tokens=("input_tokens", "sum"),
    output_tokens=("output_tokens", "sum"),
    errors=("error", lambda s: s.notna().sum()),
).round(2)
summary["accuracy"] = (summary["accuracy"] * 100).round(1)

st.subheader("Summary")
st.dataframe(summary, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Accuracy (%)")
    st.bar_chart(summary["accuracy"])
with col2:
    st.subheader("Average latency (s)")
    st.bar_chart(summary["avg_latency_s"])

col3, col4 = st.columns(2)
with col3:
    st.subheader("Average tool calls")
    st.bar_chart(summary["avg_tool_calls"])
with col4:
    st.subheader("Total tokens")
    st.bar_chart(summary[["input_tokens", "output_tokens"]])

# # pass/fail grid, one row per question, one column per model
# st.subheader("Per question results")
# grid = df.pivot_table(index="task_id", columns="model", values="correct", aggfunc="first")
# grid = grid.map(lambda v: "✅" if v else "❌")
# questions = df.drop_duplicates("task_id").set_index("task_id")["question"].str[:80]
# grid.insert(0, "question", questions)
# st.dataframe(grid, use_container_width=True)

# # drill into one question
# st.subheader("Question details")
# task = st.selectbox("Pick a question", df["task_id"].unique(),
#                     format_func=lambda t: f"{t[:8]}  {questions[t]}")
# detail = df[df["task_id"] == task]
# st.write(detail.iloc[0]["question"])
# st.dataframe(
#     detail[["model", "expected", "got", "correct", "latency_s", "tool_calls", "error"]],
#     use_container_width=True,
# )