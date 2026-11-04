"""Offline English answer-quality heuristics; no factual verification or model calls."""

from dataclasses import dataclass
import json
import re

from evaluation_config import EvaluationConfig, DEFAULT_EVALUATION_CONFIG


@dataclass(frozen=True)
class CriterionScore:
    name: str
    score: float
    reason: str


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    decision: str
    criteria: tuple[CriterionScore, ...]
    reasons: tuple[str, ...]
    acceptance_threshold: float
    blockers: tuple[str, ...]
    config: EvaluationConfig
    evaluator_version: str = "heuristic-v1"
    factual_correctness_verified: bool = False


STOPWORDS = frozenset(
    "a an the is are was were be been being to of in on at for from with by and or "
    "it its this that these those as i you your me we our can could would should "
    "do does did what who when where why how please tell give provide explain "
    "describe define list name write generate implement answer question".split()
)
UNCERTAINTY = re.compile(
    r"\bi (?:do not|don't|cannot|can't) (?:know|answer|determine|verify)\b|"
    r"\b(?:i am|i'm) (?:not sure|unsure|unable)\b|"
    r"\b(?:insufficient information|unable to answer|cannot answer|not enough information|maybe|perhaps|i think|i believe|might be)\b",
    re.IGNORECASE,
)
ERROR = re.compile(
    r"^\s*(?:error\s*:|(?:internal server error|service unavailable|"
    r"generation failed|request failed|rate limit exceeded|quota exceeded)\b)|"
    r"^\s*\{\s*\"error\"\s*:",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(r"^(?:todo|tbd|n/?a|no answer|ok|okay|sure|certainly|\[insert .*\]|\[answer\])[.! ]*$", re.IGNORECASE)
LIST_REQUEST = re.compile(
    r"\b(?:list|give|provide|name)\s+(?:exactly\s+)?"
    r"(one|two|three|four|five|six|seven|eight|nine|ten|[1-9]\d?)\b",
    re.IGNORECASE,
)
NUMBERS = dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))
CODE_REQUEST = re.compile(
    r"\b(?:write|generate|implement)\b[^.!?\n]{0,100}\b(?:code|function|script|program|sql)\b",
    re.IGNORECASE,
)
JSON_REQUEST = re.compile(r"\b(?:return|respond|output)\b[^.!?\n]{0,60}\bjson\b", re.IGNORECASE)
FACTUAL = re.compile(r"^\s*(?:what|who|when|where|is|are|does|how many|how much|calculate|compute)\b", re.IGNORECASE)


def _tokens(text):
    return re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)


def _content_terms(text):
    return set(_tokens(text)) - STOPWORDS


def _list_count(answer):
    items = re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", answer, flags=re.MULTILINE)
    if not items:
        items = re.split(r"[,;\n]+", answer)
    return len({item.strip().casefold().rstrip(".") for item in items if _tokens(item)})


def _completeness(query, answer):
    """Detect a small set of explicit output requirements, not semantic coverage."""
    scores, reasons = [1.0], []
    if answer.rstrip().endswith(("...", "…", ":")) or re.search(r"\b(?:TODO|TBD)\b", answer):
        scores.append(0.0)
        reasons.append("Answer appears unfinished or contains a placeholder.")
    requested = LIST_REQUEST.search(query)
    if requested:
        word = requested.group(1).casefold()
        count = NUMBERS[word] if word in NUMBERS else int(word)
        supplied = _list_count(answer)
        scores.append(min(supplied / count, 1.0))
        if supplied < count:
            reasons.append(f"Detected {supplied} list items; query requests {count}.")
    if CODE_REQUEST.search(query):
        code = re.search(
            r"```[^\n]*\n\s*\S|\bdef\s+\w+\s*\(|\bclass\s+\w+\s*[:{(]|"
            r"\bfunction\s+\w*\s*\(|=>|^\s*SELECT\b",
            answer, flags=re.MULTILINE,
        )
        if not code:
            scores.append(0.0)
            reasons.append("Requested code has no recognized code structure.")
    if JSON_REQUEST.search(query):
        payload = answer.strip()
        if payload.startswith("```") and payload.endswith("```"):
            payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.IGNORECASE)
        try:
            json.loads(payload)
        except (ValueError, RecursionError):
            scores.append(0.0)
            reasons.append("Requested JSON is not syntactically valid.")
    return min(scores), " ".join(reasons) or "No supported explicit requirement appears unmet."


def _structure(answer, config):
    problems = []
    if answer.count("```") % 2:
        problems.append("Unclosed code fence.")
    if "```" not in answer and any(
        len(paragraph) > config.max_paragraph_chars
        for paragraph in re.split(r"\n\s*\n", answer)
    ):
        problems.append("A paragraph exceeds the configured readability limit.")
    sentences = [s.strip().casefold() for s in re.split(r"[.!?\n]+", answer) if s.strip()]
    if len(sentences) >= 3 and len(set(sentences)) / len(sentences) < 0.5:
        problems.append("Most detected sentences are repeated.")
    return (0.0, " ".join(problems)) if problems else (1.0, "No obvious structure issue detected.")


def evaluate_answer(query, answer, *, config=DEFAULT_EVALUATION_CONFIG):
    """Return a deterministic score plus configurable acceptance blockers.

    Empty string answers are valid evaluation input and always fail. None and
    other non-string values are invalid input rather than generated answers.
    """
    if not isinstance(config, EvaluationConfig):
        raise ValueError("config must be an EvaluationConfig instance.")
    if not isinstance(query, str) or not _tokens(query):
        raise ValueError("Query must be nonempty text containing a word or number.")
    if not isinstance(answer, str):
        raise ValueError("Answer must be a text string.")
    text = answer.strip()
    present = bool(_tokens(text)) and not PLACEHOLDER.fullmatch(text)
    criteria = []
    blockers = []
    if not present:
        criteria = [CriterionScore(name, 0.0, "No meaningful answer text detected.")
                    for name in ("presence", "relevance", "completeness", "structure", "certainty", "error_free")]
        blockers.append("missing_answer")
    else:
        query_terms, answer_terms = _content_terms(query), _content_terms(text)
        overlap = query_terms & answer_terms
        if overlap:
            relevance = 1.0
            relevance_reason = f"Found {len(overlap)} shared content terms; this is lexical evidence only."
        elif FACTUAL.search(query) and len(_tokens(text)) <= config.short_answer_max_words:
            relevance = 0.5
            relevance_reason = "Concise factual-style reply; relevance cannot be established lexically."
        else:
            relevance = 0.0
            relevance_reason = "No shared content terms detected."
        completeness, completion_reason = _completeness(query, text)
        structure, structure_reason = _structure(text, config)
        uncertain = bool(UNCERTAINTY.search(text))
        error = bool(ERROR.search(text))
        criteria = [
            CriterionScore("presence", 1.0, "Answer contains non-placeholder text."),
            CriterionScore("relevance", relevance, relevance_reason),
            CriterionScore("completeness", completeness, completion_reason),
            CriterionScore("structure", structure, structure_reason),
            CriterionScore("certainty", 0.0 if uncertain else 1.0,
                           "Explicit uncertainty or inability detected." if uncertain else "No explicit inability phrase detected."),
            CriterionScore("error_free", 0.0 if error else 1.0,
                           "Response begins with a recognized failure message." if error else "No leading failure message detected."),
        ]
        if uncertain and config.fail_on_uncertainty:
            blockers.append("explicit_uncertainty")
        if error and config.fail_on_error:
            blockers.append("generation_error")
        if completeness < 1.0 and config.fail_on_incomplete:
            blockers.append("incomplete_answer")
    weights = dict(config.weights)
    score = round(sum(weights[item.name] * item.score for item in criteria) / sum(weights.values()), 6)
    passed = not blockers and score >= config.acceptance_threshold
    summary = (
        "Acceptance blocked: " + ", ".join(blockers) + "."
        if blockers else
        "Score meets the acceptance threshold." if passed else "Score is below the acceptance threshold."
    )
    return EvaluationResult(
        score, "pass" if passed else "fail", tuple(criteria),
        (summary,) + tuple(item.reason for item in criteria),
        config.acceptance_threshold, tuple(blockers), config,
    )
