"""
Streamlit chat UI with tool-use agent over extracted NHTSA risk data.

LLM picks from 5 registered tools, calls them against the loaded DataFrame,
and synthesizes a final answer (max 5 iterations per question).

Usage:
    streamlit run streamlit_app.py
"""

import json
import os
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

load_dotenv()

# Active model — matches extract_risks.py; separate free-tier quota pool from gemini-3.5-flash.
MODEL_ID = "gemini-3.5-flash-lite"
DATA_PATHS = ["output/extracted_risks.csv", "examples/sample_output.csv"]
MAX_ITERATIONS = 5


@st.cache_data
def load_data() -> tuple[pd.DataFrame | None, str | None]:
    """Return (DataFrame, source_path) for the first readable data file."""
    for path in DATA_PATHS:
        if Path(path).exists():
            df = pd.read_csv(path)
            if not df.empty:
                return df, path
    return None, None


@st.cache_resource
def get_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def make_tools(df: pd.DataFrame) -> list:
    """
    Return 5 tool functions as closures over df.
    Docstrings are written for the LLM — they tell it when to call each function.
    """

    def get_severity_distribution() -> dict:
        """USE THIS WHEN THE USER ASKS FOR THE OVERALL BREAKDOWN OR COUNT OF COMPLAINTS BY SEVERITY LEVEL. Returns a dict {critical, high, medium, low: counts}."""
        print("[TOOL] get_severity_distribution()", flush=True)
        return {k: int(v) for k, v in df["severity"].value_counts().items()}

    def get_top_makes_by_severity(severity: str, n: int = 10) -> list:
        """USE THIS WHEN THE USER ASKS WHICH MAKE OR MANUFACTURER HAS THE MOST COMPLAINTS AT A GIVEN SEVERITY LEVEL. severity must be one of: critical, high, medium, low. Returns a ranked list of makes with counts."""
        print(f"[TOOL] get_top_makes_by_severity(severity={severity!r}, n={n})", flush=True)
        severity = severity.lower().strip()
        if severity not in {"critical", "high", "medium", "low"}:
            return [{"error": f"Invalid severity: {severity}. Must be one of critical/high/medium/low."}]
        filtered = df[df["severity"] == severity]
        agg = (
            filtered.groupby("MAKETXT").size()
            .sort_values(ascending=False)
            .head(n)
            .reset_index(name="count")
        )
        return [{"MAKETXT": str(r["MAKETXT"]), "count": int(r["count"])} for _, r in agg.iterrows()]

    def get_complaints_by_component(component_keyword: str, n: int = 20) -> list:
        """USE THIS WHEN THE USER ASKS FOR COMPLAINTS INVOLVING A SPECIFIC VEHICLE SYSTEM OR PART (e.g. brake, fuel, engine, airbag). Case-insensitive substring match on the component field. Do NOT use for severity-ranked or manufacturer-count questions."""
        print(f"[TOOL] get_complaints_by_component(component_keyword={component_keyword!r}, n={n})", flush=True)
        mask = df["component"].str.contains(component_keyword, case=False, na=False)
        cols = [c for c in ["CMPLID", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
                             "severity", "failure_mode", "inferred_root_cause"] if c in df.columns]
        return json.loads(df[mask].head(n)[cols].to_json(orient="records"))

    def get_complaint_detail(cmplid: int) -> dict | None:
        """USE THIS WHEN THE USER ASKS ABOUT A SPECIFIC COMPLAINT BY ITS NUMERIC CMPLID. Returns the full row or None."""
        print(f"[TOOL] get_complaint_detail(cmplid={cmplid!r})", flush=True)
        row = df[df["CMPLID"].astype(str) == str(cmplid)]
        if row.empty:
            return None
        return json.loads(row.iloc[0].to_json())

    def search_complaints_text(keyword: str, n: int = 20) -> list:
        """USE THIS WHEN THE USER ASKS ABOUT COMPLAINTS CONTAINING SPECIFIC FREE-TEXT PHRASES (e.g. "fluid leak", "smoke", "stalls at highway speed"). Case-insensitive substring search on the raw complaint text CDESCR."""
        print(f"[TOOL] search_complaints_text(keyword={keyword!r}, n={n})", flush=True)
        if "CDESCR" not in df.columns:
            return []
        mask = df["CDESCR"].str.contains(keyword, case=False, na=False)
        cols = [c for c in ["CMPLID", "MFR_NAME", "MAKETXT", "MODELTXT", "severity"] if c in df.columns]
        subset = df[mask].head(n)[cols].copy()
        subset["CDESCR"] = df.loc[subset.index, "CDESCR"].str[:200]
        return json.loads(subset.to_json(orient="records"))

    return [
        get_severity_distribution,
        get_top_makes_by_severity,
        get_complaints_by_component,
        get_complaint_detail,
        search_complaints_text,
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def _call_model(client: genai.Client, contents: list, config) -> object:
    return client.models.generate_content(model=MODEL_ID, contents=contents, config=config)


def run_agent(
    question: str, df: pd.DataFrame, client: genai.Client
) -> tuple[str, list[dict]]:
    """
    Tool-use agent loop. Sends the question, executes any function calls the
    model returns, feeds results back, and repeats until the model returns plain
    text or MAX_ITERATIONS is reached.

    Returns (final_answer, tool_log) where each tool_log entry is
    {call: str, n: int, preview: list | dict | None}.
    """
    tools = make_tools(df)
    tool_map = {t.__name__: t for t in tools}
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a data assistant with access to tools over a NHTSA complaints dataset of 50 rows. "
            "You MUST call at least one tool to answer any factual question. "
            "Do NOT answer from your own knowledge, from memory, or from the UI sidebar — "
            "the sidebar stats are rendered to the user, NOT provided to you. "
            "If you cannot find an exact right tool, call the closest one and report what it returned."
        ),
        tools=tools,
    )

    contents: list = [types.Content(role="user", parts=[types.Part(text=question)])]
    tool_log: list[dict] = []
    response = None

    for _ in range(MAX_ITERATIONS):
        response = _call_model(client, contents, config)

        func_calls = [
            p.function_call
            for p in response.candidates[0].content.parts
            if p.function_call
        ]

        if not func_calls:
            print(f"[AGENT RETURN] tool_log={tool_log}", flush=True)
            return response.text or "", tool_log

        result_parts = []
        for fc in func_calls:
            error_str = None
            try:
                result = tool_map[fc.name](**dict(fc.args))
            except Exception as exc:
                error_str = f"{type(exc).__name__}: {exc}"
                result = {"error": error_str}

            n_rows = len(result) if isinstance(result, list) else 1
            tool_log.append({
                "name": fc.name,
                "args": dict(fc.args),
                "n_rows": n_rows,
                "preview": (result[:3] if isinstance(result, list) else result),
                "error": error_str,
            })
            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name, response={"result": result}
                    )
                )
            )

        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="user", parts=result_parts))

    last_text = (response.text or "(no text)") if response else "(no response)"
    print(f"[AGENT RETURN] max-iterations hit, tool_log={tool_log}", flush=True)
    return f"Agent hit max iterations ({MAX_ITERATIONS}). Last partial answer: {last_text}", tool_log


# ---------------------------------------------------------------------------
# Shared helper — renders tool_log entries inside a collapsed expander
# ---------------------------------------------------------------------------
def _render_tool_log(tool_log: list[dict]) -> None:
    print(f"[RENDER] received log={tool_log}", flush=True)
    with st.expander("Tool calls", expanded=False):
        if not tool_log:
            st.caption("(Tool invocations fire during each query — visible in server logs. Expander rendering under revision.)")
            return
        for entry in tool_log:
            args_str = ", ".join(f"{k}={v!r}" for k, v in entry["args"].items())
            call_str = f"{entry['name']}({args_str})"
            if entry["error"]:
                st.markdown(f"- `{call_str}` → **ERROR:** {entry['error']}")
            else:
                n_label = f"{entry['n_rows']} row" + ("s" if entry["n_rows"] != 1 else "")
                st.markdown(f"- `{call_str}` → {n_label}")
                if entry["preview"]:
                    preview = entry["preview"] if isinstance(entry["preview"], list) else [entry["preview"]]
                    st.write(preview[:3])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Supplier Risk Co-pilot", page_icon="🚗", layout="wide")

df, source = load_data()

with st.sidebar:
    st.title("Supplier Risk Co-pilot")
    st.caption("Tool-use agent over NHTSA vehicle complaints")
    st.divider()
    if df is not None:
        st.metric("Complaints loaded", len(df))
        if "MAKETXT" in df.columns:
            st.metric("Unique makes", int(df["MAKETXT"].nunique()))
        if "severity" in df.columns:
            st.subheader("Severity breakdown")
            for sev, cnt in df["severity"].value_counts().items():
                st.write(f"**{sev}**: {cnt}")
    st.divider()
    st.caption("5 tools · max 5 iterations · ~900K NHTSA records")

st.title("Supplier Risk Chat")

if df is None:
    st.error(
        "No data found. Run the extraction pipeline first:\n\n"
        "```\npython extract_risks.py --input data/sample_nhtsa.txt\n```"
    )
    st.stop()

st.caption(f"Data source: `{source}` — {len(df)} rows")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_tool_log(msg.get("tool_log", []))

# New question
if question := st.chat_input("Ask about supplier risks, components, or failure patterns …"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                final_answer, tool_log = run_agent(question, df, get_client())
            except Exception as exc:
                traceback.print_exc()
                cause = exc.last_attempt.exception() if isinstance(exc, RetryError) else exc
                final_answer = f"Agent error: {type(cause).__name__}: {cause}"
                tool_log = []

        st.markdown(final_answer)
        _render_tool_log(tool_log)

    print(f"[CHAT STORE] tool_log={tool_log}", flush=True)
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer,
        "tool_log": tool_log,
    })
