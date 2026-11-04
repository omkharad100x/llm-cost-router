"""Transparent English query complexity heuristics using only the standard library."""

from dataclasses import dataclass
import re

from complexity_config import ComplexityConfig, DEFAULT_COMPLEXITY_CONFIG


@dataclass(frozen=True)
class ComplexitySignal:
    name: str
    weight: int
    reason: str


@dataclass(frozen=True)
class ComplexityResult:
    label: str
    score: int
    signals: tuple[ComplexitySignal, ...]
    reasons: tuple[str, ...]
    word_count: int
    instruction_count: int


# Fixed rule order makes results stable even if config weights are reordered.
PATTERN_RULES = (
    ("multi_step", r"\bstep[- ]by[- ]step\b|\bfirst\b[\s\S]*\bthen\b",
     "Explicit multi-step reasoning or ordered work was requested."),
    ("reasoning", r"\b(?:prove|disprove|deduce|derive|justify)\b|\b(?:logical reasoning|counterexample|causal reasoning)\b",
     "The query requests a proof, derivation, justification, or reasoning."),
    ("coding", r"\b(?:write|generate|implement|build|create)\b[^.!?;\n]{0,100}\b(?:code|function|script|program|class|api|application|algorithm|sql|regex)\b|"
               r"(?:^|[.!?;\n]|\b(?:please|and|then))\s*(?:debug|refactor|troubleshoot)\b|"
               r"\b(?:fix|diagnose|debug)\b[^.!?;\n]{0,100}\b(?:code|bug|error|exception|traceback|function|script|program|crash)\b",
     "The query requests code generation, debugging, or refactoring."),
    ("advanced_math", r"\b(?:solve|calculate|compute|evaluate|find|differentiate|integrate)\b[^.!?;\n]{0,100}\b(?:equation|equations|integral|derivative|eigenvalue|probability|variance|matrix|quadratic|optimization)\b|"
                      r"(?:^|[.!?;\n])\s*(?:differentiate|integrate)\b|\bsolve\b[^.!?;\n]{0,100}(?:\^|=)",
     "The query requests symbolic, probabilistic, or other advanced mathematics."),
    ("analysis", r"\b(?:analy[sz]e|evaluate|compare|assess|optimi[sz]e)\b[^.!?;\n]{0,120}\b(?:trade[- ]?offs?|performance|complexity|data|dataset|strategies|approaches|risks|designs|costs|algorithms|alternatives|distribution)\b",
     "The query requests analysis, optimization, or comparison across criteria."),
    ("explanation", r"\b(?:explain in detail|detailed explanation|explain why|explain how|with examples|give examples)\b",
     "The query asks for an extended explanation or examples."),
    ("context", r"\b(?:given the following|based on the following|using the following|provided context|under the constraint|subject to)\b|```",
     "The query supplies explicit context or constraints."),
)
# Compiled below; fenced code is an explicit context signal.
RULES = tuple((name, re.compile(pattern, re.IGNORECASE), reason)
              for name, pattern, reason in PATTERN_RULES)
CLAUSE_SPLIT = re.compile(r"[.!?;\n]+|\b(?:and|then|next|finally|also)\b", re.IGNORECASE)
INSTRUCTION_START = re.compile(
    r"^(?:(?:please|first)\s+)?(?:(?:can|could|would)\s+you\s+)?"
    r"(?:explain|define|list|summarize|translate|convert|calculate|compute|write|"
    r"generate|implement|build|create|debug|fix|refactor|prove|derive|compare|"
    r"analyze|analyse|evaluate|find|solve|identify|describe|recommend|"
    r"what|who|when|where|why|how|is|are|does|do)\b",
    re.IGNORECASE,
)


def _instruction_count(text):
    clauses = {
        re.sub(r"^[\s\d)*•-]+", "", clause).strip()
        for clause in CLAUSE_SPLIT.split(text)
    }
    return sum(bool(INSTRUCTION_START.search(clause)) for clause in clauses)


def classify_query(query, *, config=DEFAULT_COMPLEXITY_CONFIG):
    """Classify one query; scores describe rule evidence, not model confidence."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a nonempty text string.")
    if not isinstance(config, ComplexityConfig):
        raise ValueError("config must be a ComplexityConfig instance.")
    text = query.strip()
    word_count = len(re.findall(r"\b\w+\b", text))
    if not word_count:
        raise ValueError("Query must contain at least one word or number.")
    instructions = _instruction_count(text)
    weights = dict(config.weights)
    evidence = []
    if word_count >= config.long_query_words:
        evidence.append(("length", f"Query has {word_count} words (limit {config.long_query_words})."))
    if instructions >= config.multiple_tasks_min:
        evidence.append(("multiple_tasks", f"Detected {instructions} distinct instruction clauses."))
    for name, pattern, reason in RULES:
        if pattern.search(text):
            evidence.append((name, reason))
    signals = tuple(ComplexitySignal(name, weights[name], reason)
                    for name, reason in evidence if weights[name] > 0)
    score = sum(signal.weight for signal in signals)
    label = "complex" if score >= config.complex_threshold else "simple"
    reasons = tuple(signal.reason for signal in signals) or (
        "No configured complexity signal was detected.",
    )
    return ComplexityResult(label, score, signals, reasons, word_count, instructions)
