"""Verify the physical contract used by PostgreSQL cache reads and writes."""
import re
from database import DatabaseError

SCOPE_TEXT = ("namespace", "embedding_model", "embedding_version", "embedding_task",
              "embedding_dtype", "eligibility_version", "cache_version", "embedding_config")
SCOPE_LIMIT = 128
KEY = ("namespace", "query_key", "embedding_model", "embedding_version", "embedding_task",
       "embedding_dimensions", "embedding_dtype", "eligibility_version", "cache_version", "embedding_config")


def _columns(cursor, table, types, c_columns):
    cursor.execute("""
        SELECT a.attname AS name, pg_catalog.format_type(a.atttypid,a.atttypmod) AS type,
               a.attnotnull AS required, (a.attgenerated='' AND a.attidentity='') AS plain,
               a.attcollation = 'pg_catalog."C"'::regcollation AS c_collation
        FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class t ON t.oid=a.attrelid
        WHERE a.attrelid=%s::regclass AND t.relkind='r' AND NOT t.relispartition
          AND a.attnum>0 AND NOT a.attisdropped
    """, (table,))
    rows = cursor.fetchall()
    if ({r["name"]:r["type"] for r in rows} != types or
            any(not r["plain"] or not r["required"] or (r["name"] in c_columns and not r["c_collation"]) for r in rows)):
        raise DatabaseError("Incompatible PostgreSQL cache columns.")


def _checks(cursor, table, expected):
    cursor.execute("""
        SELECT pg_catalog.pg_get_expr(conbin,conrelid) AS expression, convalidated AS valid
        FROM pg_catalog.pg_constraint WHERE conrelid=%s::regclass AND contype='c'
    """, (table,))
    rows = cursor.fetchall()
    def canonical(expr):
        # Each expected expression has just one comparison. PostgreSQL deparses
        # trim() using SQL TRIM syntax, and annotates the encoding literal.
        result = re.sub(r"\s+", "", expr.lower())
        result = result.replace("trim(bothfrom", "btrim(").replace("'utf8'::name", "'utf8'")
        while result.startswith("(") and result.endswith(")"):
            result = result[1:-1]
        return result

    if (not all(r["valid"] for r in rows) or
            {canonical(r["expression"]) for r in rows} != {canonical(e) for e in expected}):
        raise DatabaseError("Incompatible PostgreSQL cache check constraints.")


def _indexes(cursor, table, expected):
    cursor.execute("""
        SELECT c.contype AS kind, c.conname AS name, i.indisunique AS is_unique,
               ARRAY(SELECT a.attname::text FROM unnest(i.indkey) WITH ORDINALITY k(n,pos)
                     JOIN pg_catalog.pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.n
                     ORDER BY k.pos) AS columns,
               (i.indisvalid AND i.indisready AND i.indislive AND i.indimmediate AND
                NOT coalesce(c.condeferrable,false) AND i.indpred IS NULL AND i.indexprs IS NULL AND
                i.indnkeyatts=i.indnatts AND am.amname='btree' AND NOT EXISTS (
                    SELECT FROM unnest(i.indkey) WITH ORDINALITY k(n,pos)
                    JOIN pg_catalog.pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=k.n
                    JOIN pg_catalog.pg_opclass op ON op.oid=i.indclass[(k.pos-1)::int]
                    WHERE NOT op.opcdefault OR i.indcollation[(k.pos-1)::int] <> a.attcollation
                )) AS usable
        FROM pg_catalog.pg_index i
        JOIN pg_catalog.pg_class idx ON idx.oid=i.indexrelid
        JOIN pg_catalog.pg_am am ON am.oid=idx.relam
        LEFT JOIN pg_catalog.pg_constraint c ON c.conindid=i.indexrelid AND c.conrelid=i.indrelid
        WHERE i.indrelid=%s::regclass
    """, (table,))
    rows = cursor.fetchall()
    actual = {(r["kind"], tuple(r["columns"]), r["is_unique"]) for r in rows}
    if actual != expected or len(rows) != len(expected) or any(not r["usable"] for r in rows):
        raise DatabaseError("Incompatible PostgreSQL cache indexes.")
    return rows


def verify_catalog(cursor, version):
    cursor.execute("SELECT current_setting('server_encoding') AS encoding")
    if cursor.fetchone()["encoding"] != "UTF8":
        raise DatabaseError("PostgreSQL cache requires UTF-8.")
    meta = "lcr_cache.schema_version"
    _columns(cursor, meta, {"version":"integer"}, set())
    _checks(cursor, meta, {f"version = {version}"})
    _indexes(cursor, meta, {("p", ("version",), True)})

    table = "lcr_cache.cache_entries"
    text_columns = {"id", "query_key", "original_query", "cached_response", "response_model", *SCOPE_TEXT}
    types = {name:"text" for name in text_columns}
    types.update(embedding="bytea", embedding_dimensions="integer",
                 created_at="double precision", expires_at="double precision")
    checks = {"expires_at > created_at", "length(btrim(query_key)) > 0",
              "length(btrim(cached_response)) > 0"}
    key = KEY
    if version == 2:
        key = tuple("query_digest" if field=="query_key" else field for field in KEY)
        types["query_digest"] = "bytea"
        checks.add("query_digest = sha256(convert_to(query_key, 'UTF8'))")
        checks.update(f"octet_length({name}) <= {SCOPE_LIMIT}" for name in SCOPE_TEXT)
    _columns(cursor, table, types, {"id","query_key",*SCOPE_TEXT})
    _checks(cursor, table, checks)
    rows = _indexes(cursor, table, {("p", ("id",), True), ("u", key, True),
                                    (None, ("namespace","expires_at"), False)})
    if version == 2 and not any(r["kind"]=="u" and r["name"]=="cache_scope_digest_unique" for r in rows):
        raise DatabaseError("Unsupported PostgreSQL conflict arbiter.")
