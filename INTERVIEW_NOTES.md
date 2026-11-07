# Interview Notes — Supplier Risk Co-pilot

Quick-reference for "walk me through this choice" questions.

---

## 1. Why reservoir sampling (one-pass) instead of count-then-sample?

Two-pass would mean reading the full 1.43 GB file once to count rows, then again to sample — 2.86 GB of I/O. One-pass reservoir sampling reads the file exactly once and keeps a reservoir of *k* rows in memory, replacing each slot with probability k/i as it goes. The result is a uniform random sample with the same statistical properties, using O(k) memory instead of O(n). For a 1.43 GB file on a laptop, the difference is roughly 15 seconds vs 30 seconds and no need to know the row count in advance.

## 2. Why `response_schema` (native structured output) instead of asking for JSON?

Asking the LLM to "return JSON" and then parsing it manually introduces two failure modes: the model can produce invalid JSON (missing braces, trailing commas), and even valid JSON may have unexpected field names or types. Native structured output — `response_schema=ExtractedRisk` passed to `GenerateContentConfig` — constrains the model at the token-generation level and returns a typed Pydantic instance via `response.parsed`. There is no parsing step to fail. The only remaining failure is a null response, which we detect and retry via tenacity.

## 3. Why `ThreadPoolExecutor` + semaphore for rate limiting?

LLM calls are I/O-bound: each call spends most of its time waiting on the network. Sequential calls would waste that wait time. Using a thread pool lets multiple calls be in-flight simultaneously, cutting wall-clock time proportionally. The semaphore limits how many threads can hold an active API call at once, and each thread sleeps for `RATE_SLEEP` seconds before releasing its slot — which caps the global throughput to `MAX_CONCURRENT / RATE_SLEEP` calls per second. With `MAX_CONCURRENT=3` and `RATE_SLEEP=20`, that's 9 RPM, safely under the Gemini free-tier limit. The gap between concurrent and estimated-sequential time in the CV metrics block shows the speedup directly.

## 4. Why Pydantic v2 for the response schema?

Pydantic v2 is the validation library the Google GenAI SDK uses internally for its own schemas. Passing a Pydantic class as `response_schema` is the documented path — the SDK knows how to serialize the class into the JSON Schema the API expects and deserialize the response back into an instance. The `Literal["low", "medium", "high", "critical"]` annotation on `severity` constrains the model to exactly those four values at generation time, so there is no post-hoc normalization needed.

## 5. Why use COMPDESC as ground truth for evaluation, and why report two agreement numbers?

COMPDESC is NHTSA's own structured component tag, assigned by trained analysts who read the same complaint text the LLM reads. It is consistent and covers every row without additional labelling effort, making it a cheap proxy for a labelled test set. However, when COMPDESC is also passed to the LLM as prompt context ("reference only"), the model sometimes copies the tag verbatim into its `component` output — the pipeline saw this on rows with CMPLID 958244, where the LLM returned the COMPDESC string exactly. That inflates the agreement score to 90%, which overstates the LLM's actual extraction capability.

The `--no-compdesc-context` flag removes COMPDESC from the prompt so the LLM reads complaint text alone. That run scored 58%, which is the honest number — it reflects how well the model infers component category from raw, unstructured text with no hint. Both numbers are reported: 90% as an upper bound showing how well the pipeline performs when NHTSA context is available, and 58% as the unbiased capability baseline. In an interview, leading with 58% and explaining the methodology is more credible than leading with 90%.
