-- The runner verifies v1 and applies the whole upgrade in one transaction.
LOCK TABLE lcr_cache.cache_entries IN ACCESS EXCLUSIVE MODE;
ALTER TABLE lcr_cache.cache_entries
    ADD CONSTRAINT bound_namespace CHECK (octet_length(namespace) <= 128),
    ADD CONSTRAINT bound_embedding_model CHECK (octet_length(embedding_model) <= 128),
    ADD CONSTRAINT bound_embedding_version CHECK (octet_length(embedding_version) <= 128),
    ADD CONSTRAINT bound_embedding_task CHECK (octet_length(embedding_task) <= 128),
    ADD CONSTRAINT bound_embedding_dtype CHECK (octet_length(embedding_dtype) <= 128),
    ADD CONSTRAINT bound_eligibility_version CHECK (octet_length(eligibility_version) <= 128),
    ADD CONSTRAINT bound_cache_version CHECK (octet_length(cache_version) <= 128),
    ADD CONSTRAINT bound_embedding_config CHECK (octet_length(embedding_config) <= 128),
    ADD COLUMN query_digest bytea;
UPDATE lcr_cache.cache_entries SET query_digest = sha256(convert_to(query_key, 'UTF8'));
ALTER TABLE lcr_cache.cache_entries
    ALTER COLUMN query_digest SET NOT NULL,
    ADD CONSTRAINT digest_matches_key CHECK (query_digest = sha256(convert_to(query_key, 'UTF8'))),
    ADD CONSTRAINT cache_scope_digest_unique UNIQUE
        (namespace, query_digest, embedding_model, embedding_version, embedding_task,
         embedding_dimensions, embedding_dtype, eligibility_version, cache_version, embedding_config);

-- PostgreSQL generated the v1 constraint name; identify it by its verified columns.
DO $upgrade$
DECLARE legacy name;
BEGIN
    SELECT c.conname INTO STRICT legacy FROM pg_catalog.pg_constraint c
    WHERE c.conrelid = 'lcr_cache.cache_entries'::regclass AND c.contype = 'u'
      AND ARRAY(SELECT a.attname::text FROM unnest(c.conkey) WITH ORDINALITY AS k(n, pos)
                JOIN pg_catalog.pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.n
                ORDER BY k.pos) = ARRAY['namespace','query_key','embedding_model','embedding_version',
                    'embedding_task','embedding_dimensions','embedding_dtype','eligibility_version',
                    'cache_version','embedding_config'];
    EXECUTE format('ALTER TABLE lcr_cache.cache_entries DROP CONSTRAINT %I', legacy);
END;
$upgrade$;
ALTER TABLE lcr_cache.schema_version DROP CONSTRAINT schema_version_version_check;
UPDATE lcr_cache.schema_version SET version = 2;
ALTER TABLE lcr_cache.schema_version ADD CONSTRAINT schema_version_version_check CHECK (version = 2);
