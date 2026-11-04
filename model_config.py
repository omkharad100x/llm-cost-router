"""Explicit model-role configuration; only model-name environment variables are read."""

from dataclasses import dataclass
import os
import re


SMALL_MODEL_DEFAULT = "gemini-3.1-flash-lite"
LARGE_MODEL_DEFAULT = "gemini-3.8-flash"


@dataclass(frozen=True)
class ModelConfig:
    small_model: str = SMALL_MODEL_DEFAULT
    large_model: str = LARGE_MODEL_DEFAULT

    def __post_init__(self):
        for name in ("small_model", "large_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"gemini-[A-Za-z0-9._-]+", value):
                # Do not echo invalid environment values into errors.
                raise ValueError(f"{name} must be a Gemini model ID beginning with gemini-.")


def load_model_config(environ=None):
    """Read two model IDs explicitly; never load .env or inspect API credentials."""
    source = os.environ if environ is None else environ
    return ModelConfig(
        small_model=source.get("GEMINI_SMALL_MODEL", SMALL_MODEL_DEFAULT),
        large_model=source.get("GEMINI_LARGE_MODEL", LARGE_MODEL_DEFAULT),
    )
