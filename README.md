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

## Milestone 9: persistent semantic cache

`semantic_cache.SemanticCache` is an independent SQLite cache. It accepts the
existing `EmbeddingGenerator` and uses `semantic_similarity.find_most_similar`.
It is not integrated into the CLI, does not generate Gemini answers, and never
loads `.env` or creates a new Gemini client. No new dependencies are required.

### Eligibility is explicit and denied by default

The caller must supply a trusted assessment for **each exact query**:
`QueryEligibility(query=..., context_independent=True, non_sensitive=True,
time_independent=True, external_state_independent=True)`.
All flags default to false. Approval cannot be reused for different query text,
and truthy strings/integers do not count as true.

Saving additionally requires `ResponseEligibility(response=..., successful=True,
not_safety_refusal=True, non_sensitive=True)`, bound to the exact response.
Empty query/response text, missing approval, API failure, refusal, private content,
context dependence, and time/external-state dependence are denied. Ineligible
lookup returns a miss with reason `ineligible`; ineligible save returns `None`.
Neither operation calls the embedder for ineligible content.

**These are caller attestations, not automatic classifiers.** A trusted policy,
reviewed allowlist, or human review must establish the flags, using actual API
status/refusal information for responses. Do not set every flag to true for
arbitrary user text or let end users approve their own input. Review response
content as well as query content; a harmless question can produce private text.
No word-based heuristic can guarantee these properties. Unknown eligibility
must remain denied.

### API and behavior

```python
from semantic_cache import SemanticCache

# embedder is your existing EmbeddingGenerator(generator).
# query_review and response_review are trusted, content-bound assessments.
with SemanticCache(
    "semantic_cache.sqlite3",
    embedder,
    namespace="public-faq-policy-v1",
    embedding_version="v1",
    threshold=0.9,
    ttl_seconds=86400,
) as cache:
    result = cache.lookup(query, eligibility=query_review)
    if result.hit:
        cached_answer = result.response

    # Only after a separate application generates and reviews an answer:
    entry_id = cache.save(
        query, answer, model_used,
        eligibility=query_review,
        response_eligibility=response_review,
        ttl_seconds=3600,  # Optional per-entry override.
    )
    removed = cache.remove_expired()
    # cache.clear() explicitly clears this namespace across embedding versions.
```

`CacheLookup` contains `hit`, `response`, `matched_query`,
`similarity_score`, `entry_id`, and a `reason`. Miss payload fields are
`None`. Scores at the threshold are accepted; candidates are ordered by creation
time and ID for deterministic ties.

Lookup embeds approved input, then reads only unexpired entries with matching
namespace, embedding model/version/task/dimensions/dtype, and eligibility version.
It checks expiry after embedding latency and again before returning a hit.
Malformed stored rows are skipped; incompatible rows are excluded. If the best
entry expires during a local scan, the result is a miss. Lookup does not delete
expired rows; call `remove_expired()` to reclaim them.

Save generates an embedding for an approved pair and serializes it as a
little-endian float32 BLOB (3,072 bytes for 768 dimensions), without pickle.
Only finite normalized vectors with the expected dimensions are accepted.
The original query is preserved. An exact identity key strips only leading and
trailing whitespace; case, punctuation, and internal whitespace are retained.

An active exact duplicate with the same response and model returns its existing
ID without another embedding call or TTL refresh. An active exact query with a
different response/model raises `CacheConflictError`. An expired or corrupt
exact entry can be replaced by a fresh approved save. Paraphrases remain distinct
entries; semantic agreement between their answers is not automatically checked.

SQLite writes use parameterized SQL and short transactions. A second duplicate
check inside a write transaction handles concurrent writers after embedding.
No API request occurs while a write lock is held. Errors roll back the transaction.

### Database schema

The `cache_entries` table stores:

| Column | Purpose |
| --- | --- |
| `id` | UUID-based text primary key |
| `namespace` | Caller-defined content/generation policy scope |
| `query_key`, `original_query` | Exact identity key and original query |
| `cached_response`, `response_model` | Approved answer and generating model |
| `embedding` | Little-endian float32 BLOB |
| `embedding_model`, `embedding_version`, `embedding_task` | Compatibility metadata |
| `embedding_dimensions`, `embedding_dtype` | Vector representation |
| `eligibility_version` | Explicit-review policy schema marker |
| `created_at`, `expires_at` | UTC Unix epoch seconds |

A unique constraint covers namespace, query key, and all embedding/eligibility
metadata. An index on namespace and expiration supports filtering and cleanup.
`PRAGMA user_version = 1` identifies the schema. Nonempty unrelated databases,
unknown schema versions, or unreadable databases raise `CacheError`; they are
never reset or overwritten. The caller must provide an existing parent directory
and a dedicated cache database path.

### Limits and operation

- TTL must be finite and positive; an entry expires exactly at `expires_at`.
  Default TTL is one day. System UTC clock accuracy affects expiry.
- `remove_expired()` and `clear()` affect only the selected namespace, across its
  embedding versions, and return deleted-row counts.
- Use a context manager or `close()`. Each instance belongs to one thread.
  Separate instances can use the same database with SQLite's five-second lock timeout.
- Database failures raise sanitized `CacheError`; embedding errors propagate to the
  caller. Invalid generated vectors raise `ValueError`. No failure becomes a fake hit.
- Stored queries, answers, and embeddings are plaintext local data. Git ignores
  standard `.db`, `.sqlite`, `.sqlite3` files and their sidecars. Use one of
  these extensions. File access controls remain the application's responsibility.
- Namespaces are logical scoping, not authentication. Choose a new namespace for
  incompatible generation/prompt policies. The local `embedding_version` must
  be bumped when embedding behavior changes; it is not a discovered provider revision.
- Matching scans and copies eligible vectors in O(N * D) time/memory. There is
  no eviction by size, vector index, scheduled cleanup, or schema migration yet.
- A lookup followed by a new save embeds twice. Only active exact duplicate saves
  avoid embedding. Requests retain existing embedding latency/usage/cost instrumentation.
- Similarity does not prove answer equivalence. The default threshold needs
  calibration; explicit eligibility alone does not establish factual correctness.

Offline tests use temporary databases, deterministic clocks, and mocked embeddings.
Run the complete suite with:
`python -B -m pytest tests -q -p no:cacheprovider`.
