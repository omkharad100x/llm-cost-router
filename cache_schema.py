"""Versioned SQLite schema and transactional migration from the original cache."""

SCHEMA_VERSION = 2
LEGACY_CACHE_VERSION = "v1"
LEGACY_EMBEDDING_CONFIG = '{"normalization":"scaled-l2-float64-v1","preprocessing":"strip-v1"}'

LEGACY_SCHEMA = """
CREATE TABLE cache_entries (
    id TEXT PRIMARY KEY NOT NULL,
    namespace TEXT NOT NULL,
    query_key TEXT NOT NULL,
    original_query TEXT NOT NULL,
    cached_response TEXT NOT NULL,
    response_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding_task TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_dtype TEXT NOT NULL,
    eligibility_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    CHECK (length(trim(query_key)) > 0),
    CHECK (length(trim(cached_response)) > 0),
    CHECK (expires_at > created_at),
    UNIQUE (namespace, query_key, embedding_model, embedding_version,
            embedding_task, embedding_dimensions, embedding_dtype,
            eligibility_version)
)
"""
SCHEMA = """
CREATE TABLE cache_entries (
    id TEXT PRIMARY KEY NOT NULL,
    namespace TEXT NOT NULL,
    query_key TEXT NOT NULL,
    original_query TEXT NOT NULL,
    cached_response TEXT NOT NULL,
    response_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding_task TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_dtype TEXT NOT NULL,
    eligibility_version TEXT NOT NULL,
    cache_version TEXT NOT NULL,
    embedding_config TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    CHECK (length(trim(query_key)) > 0),
    CHECK (length(trim(cached_response)) > 0),
    CHECK (expires_at > created_at),
    UNIQUE (namespace, query_key, embedding_model, embedding_version,
            embedding_task, embedding_dimensions, embedding_dtype,
            eligibility_version, cache_version, embedding_config)
)
"""

def create_expiration_index(connection):
    connection.execute(
        "CREATE INDEX cache_expiration ON cache_entries(namespace, expires_at)"
    )


def migrate_v1(connection):
    """Called inside BEGIN IMMEDIATE; any failure rolls back all DDL and data."""
    objects = connection.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    allowed = {("table", "cache_entries"), ("index", "cache_expiration")}
    if any((row["type"], row["name"]) not in allowed for row in objects):
        raise ValueError("Custom database objects require a manual migration.")
    stored = next((row["sql"] for row in objects if row["name"] == "cache_entries"), "")
    if " ".join(stored.split()) != " ".join(LEGACY_SCHEMA.split()):
        raise ValueError("Unrecognized version 1 cache schema; migration refused.")
    columns = [row["name"] for row in connection.execute("PRAGMA table_info(cache_entries)")]
    column_sql = ", ".join(columns)  # Names from the verified legacy schema only.
    count = connection.execute("SELECT count(*) FROM cache_entries").fetchone()[0]
    connection.execute("ALTER TABLE cache_entries RENAME TO cache_entries_v1")
    connection.execute(SCHEMA)
    connection.execute(
        "INSERT INTO cache_entries (" + column_sql + ", cache_version, embedding_config) "
        "SELECT " + column_sql + ", ?, ? FROM cache_entries_v1",
        (LEGACY_CACHE_VERSION, LEGACY_EMBEDDING_CONFIG),
    )
    if connection.execute("SELECT count(*) FROM cache_entries").fetchone()[0] != count:
        raise ValueError("Migration row count mismatch.")
    connection.execute("DROP TABLE cache_entries_v1")
    create_expiration_index(connection)
    connection.execute("PRAGMA user_version = 2")
