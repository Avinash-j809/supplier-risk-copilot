"""
Streamlit chat UI over the extracted NHTSA risk data.

Loads output/extracted_risks.csv (falls back to examples/sample_output.csv),
shows sidebar stats, and answers free-form questions via Gemini using a
compact data-context prompt — no tool-use, no SQL, no function calling.

Usage:
    streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_ID = "gemini-3.5-flash-lite"
DATA_PATHS = ["output/extracted_risks.csv", "examples/sample_output.csv"]


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


def build_context(df: pd.DataFrame) -> str:
    """Compact data context for the LLM prompt: stats + 25 sampled rows."""
    sev_dist = df["severity"].value_counts().to_dict() if "severity" in df.columns else {}

    top_critical: dict = {}
    if "severity" in df.columns and "MFR_NAME" in df.columns:
        top_critical = (
            df[df["severity"] == "critical"]
            .groupby("MFR_NAME").size()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

    sample = df.sample(min(25, len(df)), random_state=42)
    keep = [c for c in ["CMPLID", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
                         "component", "failure_mode", "severity", "inferred_root_cause"]
            if c in df.columns]
    # Truncate complaint text to keep the prompt token-efficient
    if "CDESCR" in df.columns:
        sample = sample.copy()
        sample["CDESCR"] = df.loc[sample.index, "CDESCR"].str[:200]
        keep.append("CDESCR")

    return (
        f"Severity distribution: {sev_dist}\n"
        f"Top manufacturers by critical complaints: {top_critical}\n"
        f"Sample rows (CSV):\n{sample[keep].to_csv(index=False)}"
    )


def stream_answer(prompt: str):
    """Yield response text chunks; falls back to a single yield if streaming unavailable."""
    client = get_client()
    if hasattr(client.models, "generate_content_stream"):
        for chunk in client.models.generate_content_stream(model=MODEL_ID, contents=prompt):
            yield chunk.text or ""
    else:
        yield client.models.generate_content(model=MODEL_ID, contents=prompt).text or ""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Supplier Risk Co-pilot", page_icon="🚗", layout="wide")

df, source = load_data()

with st.sidebar:
    st.title("Supplier Risk Co-pilot")
    st.caption("LLM extraction over NHTSA vehicle complaints")
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
    st.caption("Source corpus: ~900K NHTSA records, 1.43 GB")

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about supplier risks, components, or failure patterns …"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    prompt = (
        "You are a supplier risk analyst reviewing NHTSA vehicle complaint data. "
        "Answer the user's question using only the data provided below. "
        "Be specific — cite CMPLIDs, manufacturer names, or component names from the data. "
        "If the data doesn't contain enough information to answer fully, say so.\n\n"
        f"=== NHTSA COMPLAINT DATA ===\n{build_context(df)}\n\n"
        f"=== USER QUESTION ===\n{question}"
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})
