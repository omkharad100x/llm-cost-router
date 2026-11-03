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
