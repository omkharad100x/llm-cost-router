# Python AI Assistant

A simple terminal-based AI assistant built with Python and Google's Gemini API.

This project was created to learn how to connect Python applications to large language models (LLMs), handle API errors, and build reliable AI-powered applications.

## Features

- Interactive question-and-answer experience
- Powered by Gemini 3.8 Flash
- Secure API key loading from environment variables
- Empty-input validation
- Basic API error handling
- Automatic retries for temporary HTTP 503 errors
- Clean exit using the `exit` command

## Tech Stack

- Python
- Google Gen AI SDK (`google-genai`)
- `python-dotenv`
- Gemini API

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/omkharad100x/python-ai-assistant
cd python-ai-assistant
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure your API key

Create a file named `.env` in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own Gemini API key. Never commit or share your actual key.

### 5. Run the assistant

```powershell
python main.py
```

Type a question to receive an answer. Type `exit` to quit.

## Project Structure

```text
python-ai-assistant/
├── .gitignore
├── main.py
├── requirements.txt
├── README.md
└── .env
```

The `.env` file is local and should be excluded from Git.

## Learning Goals

This project explores Python application structure, API integration, environment variables, error handling, retry logic, and Git/GitHub workflows.

## Future Improvements

- Conversation history and context
- Clear-conversation command
- Tool calling
- AI agent capabilities

## License

No license has been specified yet.
## Baseline request instrumentation

Every CLI answer and memory-summary generation uses `instrumentation.py`.
Each application-level attempt appends a JSON object to `request_metrics.jsonl`
beside `main.py`. The file is ignored by Git. It contains no prompt, answer,
memory, API key, or exception-message content.

Records include UTC request-start time, requested model, request kind, success/error
status, SDK call latency in milliseconds, raw prompt/candidate/thinking/total/cached
token counts, billable output tokens, estimated USD cost, and pricing version.
Latency covers the synchronous SDK invocation, including any SDK-internal retries,
but excludes record writing, CLI retry sleeps, and local conversation processing.
Each CLI retry gets its own record. An SDK error has no response usage, so its cost
is unknown even if a provider might have incurred charges.

Unavailable counts and costs are JSON `null`; a reported zero remains zero.
USD estimates are decimal strings to avoid floating-point rounding and are not
billing statements. Candidate tokens are recorded separately from thinking tokens;
both contribute to billable output. If thinking usage is absent but a consistent
total is available, billable output is inferred from total minus prompt.
If the required usage is missing, cost remains null. Unsupported tool usage and
inconsistent totals are not priced.

Edit `model_pricing.json` to configure per-model input, output, and cached-input
rates in USD per million tokens. The default is standard paid text pricing for
`gemini-3.1-flash-lite`, checked on 2026-11-03: $0.25 input, $1.50 output
(including thinking), and $0.025 cached input.
Source: https://aistudio.google.com/docs/pricing#gemini-3.1-flash-lite.
Unknown models remain unpriced. When cached usage is reported, its discount is
applied; if no cached count is supplied, the estimate uses the full input rate.
The raw cached count still remains null. Free tiers, discounts, storage, tools,
and non-text billing are outside this estimate.

A metrics-write failure emits a generic warning and preserves the provider result
or original exception. Records append without rotation; manage the local file as
needed. This is a single-process CLI recorder, not a concurrent analytics service.

Run focused offline tests with:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
```

`pytest.ini` limits default collection to `tests/`, avoiding the legacy live-call
scripts in the project root. The CLI integration tests supply synthetic memory and
mock configuration loading, user input, and Gemini calls; they do not read `.env`
or personal memory and do not make API calls.

If the pricing file cannot be read or validated, the CLI warns and continues
recording requests with unknown cost. No fallback price is silently substituted.

## Query embedding module

`embeddings.py` exposes `EmbeddingGenerator.embed_query(text)`. Supply the same
`InstrumentedGenerator` used for generation; it reuses that instance's Gemini
client (including any HTTP settings), pricing mapping, and metrics recorder.
The module never imports `main.py`, creates another client, or loads environment
variables. It is independent of the CLI and is not called by its input loop.

Example wiring inside a future application with an already configured client:

```python
from pathlib import Path

from embeddings import EmbeddingGenerator
from instrumentation import InstrumentedGenerator, JsonlRecorder, load_pricing

project_directory = Path.cwd()  # Set this to your application directory.
generator = InstrumentedGenerator(
    client,  # Your existing, configured google.genai.Client instance.
    load_pricing(project_directory / "model_pricing.json"),
    JsonlRecorder(project_directory / "request_metrics.jsonl"),
)
embedder = EmbeddingGenerator(generator)
# Explicit API request when your application calls this:
vector = embedder.embed_query("How can I reduce LLM costs?")
```

Requests use `gemini-embedding-001`, `SEMANTIC_SIMILARITY`, and 768 output
dimensions. NumPy normalization returns a unit-length float64 array, so the dot
product of two returned vectors is their cosine similarity. Scaling before the
norm prevents numeric overflow/underflow. All vectors intended for comparison
must use this same model, task type, and dimension.

Empty or non-string input raises `ValueError` before any request. Missing,
multiple, truncated, incorrectly sized, nonnumeric, nonfinite, or zero vectors
raise `EmbeddingResponseError`. API exceptions propagate unchanged so callers
can decide how to retry; this module adds no retries.

Embedding requests use the same JSONL instrumentation and log no text or vector.
The installed SDK 2.29.0 exposes optional token usage as
`response.embeddings[0].statistics.token_count`, typed as a float. Only finite,
nonnegative integral counts are accepted. This field is documented primarily for
the Vertex backend and may be absent for Gemini Developer API responses. Missing
usage/cost remains null; a reported zero remains zero. Character counts are not
converted to tokens, and no extra count-tokens call is made.

The configurable standard text embedding estimate is $0.15 per million input
tokens, checked 2026-11-03:
https://aistudio.google.com/docs/pricing#gemini-embedding.
The pricing entry's zero output rate describes input-only billing; the record's
`output_tokens` remains null because the SDK does not report output token usage.

Latency and status describe the SDK invocation, excluding normalization. A
successful SDK invocation with an invalid vector is recorded as success before
local validation rejects it; reported usage is still counted. An API exception is
recorded as error. Recorder failures warn without hiding API or validation errors.
No live embedding request is performed during imports or offline tests.

## Semantic similarity search

`semantic_similarity.py` searches an in-memory mapping of stored query strings
to normalized vectors. Supply vectors returned by `EmbeddingGenerator.embed_query`:

```python
from semantic_similarity import find_most_similar

# query_vector and stored_vector were generated previously by embeddings.py.
match = find_most_similar(
    query_vector,
    {"How can I reduce LLM costs?": stored_vector},
    threshold=0.9,
)
if match is not None:
    print(match.query, match.score)
```

The function returns a frozen `SimilarityMatch(query, score)`, or `None` for
an empty collection or when the best score is below the threshold. The default
threshold is 0.9; any finite numeric value from -1 to 1 is supported. Equality
passes the threshold. Ties select the first entry in mapping iteration order.

Vectors must be finite, one-dimensional, nonempty, numeric, and normalized.
All candidates must match the query's dimensions. Production vectors from
`embeddings.py` have 768 dimensions; smaller consistent dimensions are accepted
for independent use. Norm drift within `rtol=1e-6, atol=1e-8` is corrected on
copies; arbitrary unnormalized vectors, zero vectors, malformed collections,
and invalid keys or thresholds raise `ValueError`. No invalid entry is silently
skipped. Identical/opposite score endpoints are stabilized within eight float64
machine epsilons; interior threshold comparisons have no tolerance.

NumPy computes all candidate scores with one matrix-vector multiplication.
Each call validates and copies the candidates, taking O(N * D) time and temporary
memory. There is no persistent index, API call, or storage access. The caller
must ensure compatible model/task settings; vector values alone cannot verify
that. Similarity does not guarantee answer equivalence, and the default threshold
has not been calibrated on a labeled dataset.
