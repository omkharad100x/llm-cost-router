"""Explicit database settings; never load environment files."""
from dataclasses import dataclass, field
import math
import os
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class DatabaseConfig:
    mode: str = "development"
    path: Path = field(default=ROOT / "cache.sqlite3", repr=False)
    timeout_seconds: float = 5.0

    def __post_init__(self):
        if self.mode not in ("development", "production"):
            raise ValueError("Database mode must be development or production.")
        if not isinstance(self.path, Path) or self.path.suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
            raise ValueError("Database path must name a SQLite file.")
        if not self.path.is_absolute() or str(self.path).startswith(("\\\\", "//")):
            raise ValueError("Database path must be absolute and local.")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or not 0 <= self.timeout_seconds <= 60):
            raise ValueError("Database timeout must be between 0 and 60 seconds.")

@dataclass(frozen=True)
class PostgresConfig:
    dsn: str = field(repr=False)
    mode: str = "production"

    def __post_init__(self):
        try:
            url = urlsplit(self.dsn)
            options = parse_qsl(url.query, strict_parsing=True, keep_blank_values=True)
            if (self.mode not in ("development", "production")
                    or url.scheme not in ("postgres", "postgresql") or not url.hostname
                    or not re.fullmatch(r"[A-Za-z0-9.:-]+", url.hostname)
                    or any(char.isspace() or ord(char) < 32 for char in self.dsn)
                    or not url.username or not url.password or url.path in ("", "/")
                    or url.fragment or (url.port is not None and url.port < 1)
                    or len(options) != len(dict(options))
                    or any(k not in ("sslmode", "sslrootcert") or not v for k, v in options)
                    or dict(options).get("sslmode", "verify-full") != "verify-full"):
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Invalid PostgreSQL configuration.") from None


def load_database_config(environ=None):
    source = os.environ if environ is None else environ
    mode = source.get("LCR_DATABASE_MODE", "development")
    backend = source["LCR_DATABASE_BACKEND"] if "LCR_DATABASE_BACKEND" in source else "sqlite"
    if backend == "postgresql":
        return PostgresConfig(source.get("LCR_POSTGRES_DSN", ""), mode)
    if backend != "sqlite":
        raise ValueError("Unsupported database backend.")
    raw_path = source.get("LCR_DATABASE_PATH")
    raw_timeout = source.get("LCR_DATABASE_TIMEOUT_SECONDS", "5")
    if mode == "production" and not raw_path:
        raise ValueError("Production requires an explicit database path.")
    try:
        if raw_path is not None and (not isinstance(raw_path, str) or not raw_path.strip()
                                    or "://" in raw_path or raw_path.startswith("file:")):
            raise ValueError
        if not isinstance(raw_timeout, str):
            raise ValueError
        return DatabaseConfig(mode=mode, path=Path(raw_path) if raw_path is not None else ROOT/"cache.sqlite3",
                              timeout_seconds=float(raw_timeout))
    except (TypeError, ValueError):
        raise ValueError("Invalid database configuration.") from None

def database_summary(environ=None):
    """Configuration only: importing or displaying this never opens a database."""
    try:
        config = load_database_config(environ)
        return dict(backend="PostgreSQL" if isinstance(config, PostgresConfig) else "SQLite", mode=config.mode, configuration="Configured", connection="Not probed")
    except ValueError:
        return dict(backend="Unavailable", mode="Unavailable", configuration="Invalid", connection="Not probed")
