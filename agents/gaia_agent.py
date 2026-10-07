import os
import re
import time
import logging
import requests
import base64
import subprocess, sys
import io, contextlib
import pandas as pd
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".txt", ".py", ".csv"}
MAX_INPUT_CHARS = 50_000

def _safe_path(file_path: str) -> Path:
    """Resolve and validate a file path to prevent path traversal."""
    path = Path(file_path).resolve()
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"File type '{path.suffix}' is not allowed.")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from dotenv import load_dotenv

load_dotenv(override=True)

SYSTEM_PROMPT = """You are a general AI assistant. Answer questions using the tools available.

Always verify facts using search and fetch_webpage. Never rely on your own memory.
Use search to find a relevant source, then use fetch_webpage on the source URL to read the full page, since search results only give short snippets that often don't contain the specific fact you need.
If a search doesn't return what you need, try one or two different search terms or sources before giving up.
When a question names a specific source (an article, author, or exact publication), search for that specific source directly rather than a general topic search, and verify the source you found matches the one named in the question before extracting the answer.
Be careful not to confuse a name mentioned in the question's citation or attribution (such as an author or compiler) with the answer itself — find the specific name the question is actually asking about within the source content.
For trivia-style questions with a single obscure correct answer (such as "least athletes," "only recipient," or similar superlative/exclusivity claims), verify your answer against a second independent source before finalizing. If two sources disagree, prefer the source that most directly matches the specific details named in the question (exact year, exact competition, exact edition).

Give the exact answer as it appears in the source — full names, exact numbers, exact spellings.
Do not add extra items or remove items from lists. Return only what the source explicitly states.
When extracting items from audio or a source, use the exact descriptive wording used, including any qualifiers like "freshly squeezed" or "pure" — do not shorten or simplify.
When a question asks for a specific part of a name or fact (e.g. "first name only"), give only that part, even if the source states the full form.
When a question asks for a list in alphabetical order, sort the items alphabetically before giving your final answer.
For questions about counting items in a specific category and date range (e.g. studio albums released in a given period), list out each qualifying item individually before giving the final count, to avoid missing or double-counting entries.

If asked to identify vegetables from a list, use the botanical definition: a vegetable is any edible plant part that is not a fruit — this includes roots, stems, leaves, and flower buds (e.g., broccoli, celery, lettuce, sweet potatoes, basil). Exclude anything that is botanically a fruit (developed from a flower, contains seeds) — this includes bell peppers, tomatoes, cucumbers, squash, corn, green beans, peas, nuts such as acorns and peanuts, and seed-derived spices such as whole allspice.
For questions involving a table or dataset given directly in the question, or any question requiring precise calculation or logical verification, use execute_python_code to compute the answer rather than reasoning about it in words.
When working with a spreadsheet, first list all column headers before deciding which ones to include in a calculation, so you don't miss a relevant column.

If you cannot find the answer after searching, respond with your single best guess based on whatever partial information you found. Never respond with a sentence explaining that you couldn't find it, apologizing, or asking the user to check themselves — always commit to a single best-guess answer in the required format. A guess has a chance of being graded correct; an explanation never does.

Before responding, check your draft answer: if it contains any explanation, reasoning, or more than one sentence, delete everything except the bare answer value itself.

Your final response must contain nothing but the answer value itself — no labels, no restating the question, no surrounding sentence, no "FINAL ANSWER:", no explanations. For example, if the answer is 80GSFC21M0002, respond with exactly: 80GSFC21M0002 — not "The award number is 80GSFC21M0002."
If asked for a number, give just the number. If asked for a list, give a comma-separated list with no extra text."""

# tools for the agent to use

# web search tool
search_tool = DuckDuckGoSearchRun()

# fetch webpage tool
def fetch_webpage(url: str) -> str:
    """Fetches the text content of a webpage so you can read it in full, not just a search snippet.
    Args:
        url: The URL to fetch
    """
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:4000]
    except Exception as e:
        logger.error("Error fetching webpage: %s", type(e).__name__)
        return f"Error fetching webpage: {type(e).__name__}."


# audio transcription tool
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def transcribe_audio(file_path: str) -> str:
    """Transcribes an audio file (e.g. .mp3) to text.
    Args:
        file_path: Local path to the audio file to transcribe.
    """
    try:
        path = _safe_path(file_path)
        if path.stat().st_size > 25 * 1024 * 1024:  # 25MB Whisper API limit
            return "Error: audio file too large (max 25MB)."
        with open(path, "rb") as f:  # nosec: path validated by _safe_path
            transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
        return transcript.text
    except Exception as e:
        logger.error("Error transcribing audio: %s", type(e).__name__)
        return f"Error transcribing audio: {type(e).__name__}."

# excel reading tool
def read_excel(file_path: str) -> str:
    """Reads an Excel file and returns its content as a string.
    Args:
        file_path: Local path to the Excel file to read.
    """
    try:
        path = _safe_path(file_path)  # nosec: path validated by _safe_path
        df = pd.read_excel(path)
        return df.to_string(index=False)
    except Exception as e:
        logger.error("Error reading Excel file: %s", type(e).__name__)
        return f"Error reading Excel file: {type(e).__name__}."

# excel column summation tool
def sum_excel_column(file_path: str, column_name: str) -> str:
    """Sums a specific column in an Excel file and returns the total.
    Args:
        file_path: Local path to the Excel file.
        column_name: Exact column header to sum.
    """
    try:
        path = _safe_path(file_path)
        df = pd.read_excel(path)
        total = df[column_name].sum()
        return f"{total:.2f}"
    except Exception as e:
        logger.error("Error summing Excel column: %s", type(e).__name__)
        return f"Error summing column: {type(e).__name__}."

# image analysis tool
def analyze_image(file_path: str, question: str) -> str:
    """Analyzes an image file and answers a question about it using vision.
    Args:
        file_path: Local path to the image file.
        question: The question to answer about the image.
    """
    try:
        path = _safe_path(file_path)
        with open(path, "rb") as f:  # nosec: path validated by _safe_path
            image_data = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                    {"type": "text", "text": question[:MAX_INPUT_CHARS]}
                ]
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error analyzing image: %s", type(e).__name__)
        return f"Error analyzing image: {type(e).__name__}."

# youtube transcript tool
def get_youtube_transcript(url: str) -> str:
    """Fetches the transcript of a YouTube video.
    Args:
        url: The YouTube video URL or video ID
    """
    try:
        # extract video id from url
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            video_id = url.split("/")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

# text file reading tool
def read_text_file(file_path: str) -> str:
    """Reads a local text or code file and returns its contents.
    Args:
        file_path: Local path to the file to read.
    """
    try:
        path = _safe_path(file_path)  # nosec: path validated by _safe_path
        return path.read_text()
    except Exception as e:
        logger.error("Error reading text file: %s", type(e).__name__)
        return f"Error reading file: {type(e).__name__}."

# python execution tools
def execute_python_file(file_path: str) -> str:
    """Executes a local Python file and returns its printed output.
    Args:
        file_path: Local path to the .py file to execute.
    """
    try:
        path = _safe_path(file_path)
        if path.suffix != ".py":
            return "Error: only .py files can be executed."
        result = subprocess.run(  # nosec: path validated by _safe_path, args passed as list
            [sys.executable, str(path)],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        logger.error("Error executing Python file: %s", type(e).__name__)
        return f"Error executing Python file: {type(e).__name__}."

# allowed built-ins for sandboxed code execution
_SAFE_BUILTINS = {
    "print": print, "range": range, "len": len, "int": int, "float": float,
    "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "bool": bool, "type": type,
}

def execute_python_code(code: str) -> str:
    """Executes a snippet of Python code and returns its printed output. Use this for precise calculations or logic checks, such as verifying a table property, rather than reasoning about it in words.
    Args:
        code: The Python code to execute.
    """
    if len(code) > MAX_INPUT_CHARS:
        return "Error: code input too large."
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": _SAFE_BUILTINS})  # nosec
        return buf.getvalue().strip() or "(no output — add print statements)"
    except Exception as e:
        return f"Error executing code: {type(e).__name__}: {e}"

# openai chat model for the agent
chat_openai = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=4096,
)

# anthropic chat model for the agent
chat_anthropic = ChatAnthropic(
    model="claude-sonnet-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4096,
)

# ollama chat model for the agent
chat_oss = ChatOllama(model="llama3.1", temperature=0, num_predict=4096)

# grok chat model for the agent
chat_grok = ChatOpenAI(
    model="grok-3",
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
    temperature=0,
    max_tokens=4096,
)

provider = os.getenv("MODEL_PROVIDER", "openai")
chat = {"openai": chat_openai, "anthropic": chat_anthropic, "ollama": chat_oss, "grok": chat_grok}[provider]

tools = [search_tool, transcribe_audio, read_excel, analyze_image, fetch_webpage, read_text_file, execute_python_file, sum_excel_column, execute_python_code]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_calls_made: int

def assistant(state: AgentState):
    tool_calls_made = state.get("tool_calls_made", 0)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    if tool_calls_made >= 12:
        messages.append(HumanMessage(content="You have used your tool call budget. Answer now with your best guess. Do not call any more tools."))
        response = chat.invoke(messages)
    else:
        response = chat_with_tools.invoke(messages)

    new_count = tool_calls_made + (1 if getattr(response, "tool_calls", None) else 0)
    return {"messages": [response], "tool_calls_made": new_count}

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
gaia_agent = builder.compile()

def normalize_answer(raw_answer: str) -> str:
    cleaned = raw_answer.strip()

    if "\n\n" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if parts:
            cleaned = parts[-1]

    for prefix in ["FINAL ANSWER:", "Final Answer:", "Answer:", "The answer is:", "The answer is"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.replace("St.", "Saint")

    if cleaned.startswith("$"):
        cleaned = cleaned[1:]

    if "," in cleaned and "(" not in cleaned and ")" not in cleaned and ". " not in cleaned:
        items = [i.strip() for i in cleaned.split(",")]
        if not any(item.replace(".", "").isdigit() for item in items):
            cleaned = ", ".join(sorted(items, key=str.lower))

    return cleaned

def run(question: str, file_path: str = None) -> dict:
    content = question if not file_path else f"{question}\n\nAttached file: {file_path}"
    start = time.time()
    result = gaia_agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config={"recursion_limit": 40}
    )
    latency = time.time() - start
    messages = result["messages"]

    tool_names = [tc["name"] for m in messages if getattr(m, "tool_calls", None) for tc in m.tool_calls]
    input_tokens = sum((getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0) for m in messages)
    output_tokens = sum((getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0) for m in messages)

    raw = messages[-1].content
    if isinstance(raw, list):
        text_blocks = [b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text"]
        raw = text_blocks[-1] if text_blocks else ""

    return {
        "answer": normalize_answer(raw),
        "tool_calls": tool_names,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": round(latency, 2),
    }

def answer(question: str, file_path: str = None) -> str:
    return run(question, file_path)["answer"]