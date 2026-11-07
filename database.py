"""Opt-in SQLite and PostgreSQL lifecycle; no clients or connections at import."""
from contextlib import closing
from datetime import datetime, timezone
import sqlite3

from cache_schema import SCHEMA, SCHEMA_VERSION
from database_config import DatabaseConfig, PostgresConfig, load_database_config

class DatabaseError(RuntimeError):
    """Safe database error without paths, credentials or raw SQLite details."""

def _config(config):
    result = load_database_config() if config is None else config
    if not isinstance(result, (DatabaseConfig, PostgresConfig)):
        raise ValueError("Expected DatabaseConfig.")
    return result

def _path(config):
    path = config.path
    try:
        if path.resolve() != path or path.is_symlink():
            raise ValueError
        if not path.parent.is_dir():
            raise ValueError
    except (OSError, ValueError, RuntimeError):
        raise DatabaseError("Database location is unavailable or linked.") from None
    return path

def _validate_schema(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        raise DatabaseError("Database schema requires an explicit initialization or migration.")
    row = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='cache_entries'").fetchone()
    if row is None or " ".join(row[0].split()) != " ".join(SCHEMA.split()):
        raise DatabaseError("Database cache schema is incompatible.")
    indexes = connection.execute("PRAGMA index_info(cache_expiration)").fetchall()
    if [row[2] for row in indexes] != ["namespace", "expires_at"]:
        raise DatabaseError("Database cache index is incompatible.")

def open_cache(embedder, *, database_config=None, **cache_options):
    """Open a context-managed cache using only the selected backend."""
    config = _config(database_config)
    if isinstance(config, PostgresConfig):
        from postgres_cache import PostgresCache
        return PostgresCache(config, embedder, **cache_options)
    path = _path(config)
    # Import lazily so dashboard configuration checks have no model-layer imports.
    from semantic_cache import SemanticCache, CacheError
    cache = None
    try:
        cache = SemanticCache(path, embedder, connection_timeout=config.timeout_seconds,
                              require_existing=config.mode == "production",
                              allow_initialize=config.mode != "production", **cache_options)
        _validate_schema(cache._db())
        cache._db().execute("PRAGMA synchronous=FULL")
        return cache
    except (sqlite3.Error, CacheError, DatabaseError):
        if cache is not None:
            cache.close()
        raise DatabaseError("Cache database could not be opened or validated.") from None

def initialize_database(config=None):
    """Initialize a new SQLite file or apply the versioned PostgreSQL migration."""
    config = _config(config)
    if isinstance(config, PostgresConfig):
        from postgres_cache import migrate
        return migrate(config)
    path = _path(config)
    try:
        with path.open("xb"):
            pass
    except OSError:
        raise DatabaseError("Database initialization requires a new writable file.") from None
    from semantic_cache import SemanticCache, CacheError
    try:
        with SemanticCache(path, None, connection_timeout=config.timeout_seconds, require_existing=True):
            pass
    except (sqlite3.Error, CacheError):
        # Preserve failed files for inspection. Never delete or overwrite data.
        raise DatabaseError("Database initialization failed; inspect the file before retrying.") from None

def verify_database(config=None):
    """Read-only probe: no schema migration, creation, generation or cached text."""
    config = _config(config)
    if isinstance(config, PostgresConfig):
        from postgres_cache import verify
        return verify(config)
    path = _path(config)
    try:
        with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True,
                                     timeout=config.timeout_seconds)) as connection:
            connection.execute("PRAGMA query_only=ON")
            _validate_schema(connection)
            if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise DatabaseError("Database integrity verification failed.")
        return dict(backend="SQLite", connection="Verified", schema_version=SCHEMA_VERSION,
                    checked_at_utc=datetime.now(timezone.utc).isoformat())
    except (sqlite3.Error, DatabaseError):
        raise DatabaseError("Database verification failed.") from None
