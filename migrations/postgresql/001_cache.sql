CREATE SCHEMA lcr_cache;
REVOKE ALL ON SCHEMA lcr_cache FROM PUBLIC;
CREATE TABLE lcr_cache.schema_version (
    version INTEGER PRIMARY KEY CHECK (version = 1)
);
CREATE TABLE lcr_cache.cache_entries (
    id TEXT COLLATE "C" PRIMARY KEY NOT NULL,
    namespace TEXT COLLATE "C" NOT NULL,
    query_key TEXT COLLATE "C" NOT NULL,
    original_query TEXT NOT NULL,
    cached_response TEXT NOT NULL,
    response_model TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    embedding_model TEXT COLLATE "C" NOT NULL,
    embedding_version TEXT COLLATE "C" NOT NULL,
    embedding_task TEXT COLLATE "C" NOT NULL,
    embedding_dimensions INTEGER NOT NULL,
    embedding_dtype TEXT COLLATE "C" NOT NULL,
    eligibility_version TEXT COLLATE "C" NOT NULL,
    cache_version TEXT COLLATE "C" NOT NULL,
    embedding_config TEXT COLLATE "C" NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    CHECK (length(trim(query_key)) > 0),
    CHECK (length(trim(cached_response)) > 0),
    CHECK (expires_at > created_at),
    UNIQUE (namespace, query_key, embedding_model, embedding_version,
            embedding_task, embedding_dimensions, embedding_dtype,
            eligibility_version, cache_version, embedding_config)
);
CREATE INDEX cache_expiration ON lcr_cache.cache_entries(namespace, expires_at);
REVOKE ALL ON ALL TABLES IN SCHEMA lcr_cache FROM PUBLIC;
INSERT INTO lcr_cache.schema_version(version) VALUES (1);
