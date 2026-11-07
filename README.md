# Supplier Risk Co-pilot

This project extends a prior [Automobile Supply Chain Risk Analytics](https://github.com/avinash/automobile-supply-chain-risk) work (300 suppliers, 900K NHTSA records, Python + SQL Server + Power BI) by adding an LLM extraction layer over unstructured vehicle-complaint text. It samples 200 rows from the 1.43 GB NHTSA flat file, calls Gemini concurrently for structured risk extraction (component, failure mode, severity, root-cause hypothesis), and evaluates LLM output against NHTSA's own ground-truth component tags — producing CV-ready metrics without any labelled data.

## Architecture

```
FLAT_CMPL.txt (1.43 GB)
        │
        ▼
  Reservoir Sampling              ← one-pass O(n), O(k) memory, random.seed(42)
  200 rows × 8 columns
        │
        ▼
  pandas DataFrame
        │
        ├─ CDESCR (free text) ──► Gemini 3.5 Flash  ◄── response_schema=ExtractedRisk
        └─ COMPDESC (ground truth)   (concurrent, rate-limited)
                                          │
                                          ▼
                                  ExtractedRisk
                                  ├── component
                                  ├── failure_mode
                                  ├── severity
                                  └── inferred_root_cause
                                          │
                                          ▼
                              Agreement score vs COMPDESC
                                          │
                                          ▼
                              output/extracted_risks.csv
                              + CV-ready metrics block
```

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your Gemini API key:
#   GEMINI_API_KEY=your_key_here
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/).

## Run

**Full pipeline (real NHTSA file):**
```bash
python extract_risks.py --input "D:\Projects\FLAT_CMPL.txt"
```

**Sample file — works out of the box, no 1.43 GB download needed:**
```bash
python extract_risks.py --input data/sample_nhtsa.txt
```

**Synthetic fallback — no input file required:**
```bash
python extract_risks.py --input nonexistent.txt
```

Output is written to `output/extracted_risks.csv`.

`data/sample_nhtsa.txt` contains real rows from the NHTSA corpus (first 50 lines of `FLAT_CMPL.txt`).

> **`examples/sample_output.csv` — not yet committed.** The file currently in `examples/` is labelled `sample_output_SYNTHETIC_DO_NOT_SHIP.csv` and was produced from the synthetic fallback during development; it does not represent real extraction output. Once the Gemini API daily quota resets, run `python extract_risks.py --input data/sample_nhtsa.txt` to generate real output and commit the first 20 rows as `examples/sample_output.csv`.

> **Free-tier note:** Gemini API free tier has a daily request quota (~50–100 calls/day depending on model). The pipeline is designed for the full 200-row run; use `data/sample_nhtsa.txt` (50 rows) for a quick demo that stays within quota.

## Built on

| Metric | Value |
|---|---|
| Rows processed | 50 real NHTSA complaints |
| Source corpus | ~900K complaints, 1.43 GB |
| Fields extracted | 4 (component, failure_mode, severity, inferred_root_cause) |
| LLM-COMPDESC agreement (with COMPDESC hint) | 90.0% — upper bound; LLM can see the ground-truth tag |
| LLM-COMPDESC agreement (CDESCR only, unbiased) | **58.0%** — honest number; LLM reads complaint text only |
| Wall-clock (3 concurrent) | ~364–374s for 50 rows at ~9 RPM |
| Failures | 0 / 50 |
| Model | gemini-3.5-flash-lite, native structured output via `response_schema` |

**Why two numbers?** When COMPDESC is passed as context, the model sometimes copies the ground-truth tag verbatim into its `component` output, inflating the agreement score. The unbiased run (flag `--no-compdesc-context`) removes COMPDESC from the prompt entirely, so the 58% reflects how well the LLM infers component from raw complaint text alone — a more honest capability benchmark.

*Output is from a 200-row sample of the ~900K-row corpus. Full corpus processing would require a paid API tier.*

## Roadmap

- **Streamlit chat UI** — natural-language queries over extracted risk data
- **Tool-use agent for SQL** — agentic loop that queries structured risk tables and explains results
- **GitHub Actions weekly digest** — LLM-written narrative email summarizing new high-severity complaints per supplier
- **Jira ticket auto-creation** — automatically opens tickets for critical-severity clusters above a threshold
- **Full corpus ingestion** — batch processing of all 900K complaints with paid API tier

## Prior work

This builds on [Automobile Supply Chain Risk Analytics](#) — a Power BI + SQL Server dashboard tracking defect trends across 300 suppliers using 900K NHTSA complaints. This project adds the LLM extraction layer on top of that raw data to convert unstructured complaint text into structured risk signals.
