# Cache database configuration

## SQLite local configuration and Task 7 history

Task 7 supported Python's built-in SQLite on one server with a durable local
volume. SQLite remains the default local backend. Task 8A adds the PostgreSQL
adapter for the selected Supabase Free transaction-pooler target. The cache
already uses SQLite transactions, schema version 2, float32 embedding blobs and
namespace/version filters. Reusing that tested implementation avoids a premature
second storage backend.

Before Task 7, callers selected an arbitrary SQLite path directly. Cache records
contained reviewed queries, responses, embeddings, model/configuration metadata,
IDs and TTL timestamps. Request metrics remain JSONL; benchmark results remain
JSON. Neither dashboard reads cache records, and the CLI is still independent
of semantic caching. Task 7 adds an opt-in configured cache factory, explicit
initialization, read-only verification and configuration indicators.

| Consideration | Selected SQLite configuration | Managed PostgreSQL alternative |
| --- | --- | --- |
| Persistence | Durable local volume required | Remote database independent of app files |
| Concurrent access | Multiple connections; one writer at a time | Better suited to multiple writers/hosts |
| Multiple app instances | Same host only; not a shared network file | Appropriate for multiple app hosts |
| Connections | One connection per cache instance/thread; context-managed | Driver plus bounded per-process pools and total connection budget |
| Transactions/migrations | Existing transactional v1-to-v2 migration retained | New dialect, schema migrations and conflict handling required |
| Hosting | Host must retain the volume across restarts/redeploys | Suitable when app hosts have ephemeral storage |
| Operations | File permissions, storage capacity and backup procedures | Credentials, TLS, provider limits, pooling and backup configuration |
| Cost | No external database service introduced; host, disk and backups may cost money | Provider plan required; no paid service or free-tier entitlement assumed |
| Recovery | Operator-managed backup and tested restore | Provider-specific backup/PITR features must be configured and verified |

SQLite recommends a client/server database for many concurrent writers or
multiple servers; network filesystem locking is not an appropriate shortcut.
See [SQLite deployment guidance](https://sqlite.org/whentouse.html) and
[network filesystem cautions](https://sqlite.org/useovernet.html).

PostgreSQL configuration and migration instructions follow the SQLite section.
No provider account, paid plan or production database was provisioned.

## Environment configuration

SQLite uses these process environment variables. No dotenv loading occurs.

| Variable | Default | Production requirement |
| --- | --- | --- |
| LCR_DATABASE_MODE | development | Set to production |
| LCR_DATABASE_PATH | Repository-root cache.sqlite3 | Explicit absolute local .db/.sqlite/.sqlite3 path |
| LCR_DATABASE_TIMEOUT_SECONDS | 5 | Finite number from 0 to 60; SQLite busy timeout |

Invalid settings fail with safe errors rather than falling back to a default.
URI/credential connection strings, relative paths and UNC paths are rejected.
Configuration summaries and object representations omit the database path.
Mapped/network mounts cannot be reliably identified by the application: the
operator must ensure the selected path is on a supported local persistent volume.
This configuration does not establish OS permissions or defend against a
privileged process swapping filesystem paths.

Example Windows settings (paths contain no credentials):

    $env:LCR_DATABASE_MODE = "production"
    $env:LCR_DATABASE_PATH = "D:\LlmRouterData\cache.sqlite3"
    $env:LCR_DATABASE_TIMEOUT_SECONDS = "5"

For a Linux host, use an absolute path such as /var/lib/llm-cost-router/cache.sqlite3.
Provision that parent directory and volume during deployment. The application
does not silently create missing production directories.

## Initialization, verification and use

From the repository root, explicitly initialize a **new** configured file:

    .\.venv\Scripts\python.exe -B database_admin.py initialize

Initialization refuses any existing file, even an empty one. It never overwrites
data. If initialization fails after file creation, the file is retained for
inspection; do not blindly delete or retry against an existing database.

For an existing database, verify without initialization or migration:

    .\.venv\Scripts\python.exe -B database_admin.py verify

The verifier opens SQLite in read-only mode, checks schema version, table
constraints, expiration index and quick_check, then closes the connection.
It returns a timestamped success or a generic failure with a nonzero exit code.
A verified read connection does not prove write permission, durable storage,
backup readiness or future availability.

Callers can use the existing embedding generator through the configured factory:

    from database import open_cache
    from semantic_cache import QueryEligibility

    # embedder is the application's existing injected EmbeddingGenerator.
    with open_cache(embedder, namespace="my-application") as cache:
        result = cache.lookup(query, eligibility=review)

Production opening uses SQLite mode=rw: missing files cannot be recreated.
Existing empty databases are refused. Development mode can initialize its
configured path, preserving the local workflow. Direct SemanticCache callers
retain their existing defaults. The factory checks the schema and expiration
index and sets synchronous=FULL; it does not change the journal mode or add
automatic retries.

A cache instance must stay on its creating thread. Use context managers and
separate instances for concurrent workers; do not put a single connection in
a global cross-thread dashboard cache. SQLite needs no network connection pool.
Initialization/opening can acquire a write lock for schema verification or
migration; lock failures raise a safe DatabaseError. No alternate database or
in-memory fallback is attempted.

## Existing data, migration and recovery

No production database was opened or migrated during implementation. Offline
tests use disposable databases with production mode enabled.

For existing SQLite data:

1. Stop cache writers and retain the original database.
2. Create a consistent backup using SQLite's backup API; do not copy an active
   database file while ignoring journal/WAL sidecars. Protect the backup like
   the source. See the [SQLite backup documentation](https://www.sqlite.org/backup.html).
3. Restore a copy onto the durable volume and verify its rows and metadata.
4. Schema version 2 is checked by verify. A recognized version 1 requires
   opening through open_cache to run the existing transactional migration;
   perform this against the restored copy before serving requests. verify never
   migrates. Unsupported schemas are refused.
5. Reopen and verify the migrated copy, compare row counts and metadata to the
   original, and exercise namespace isolation before switching the configured path.
6. Keep the original and rollback configuration until acceptance is complete.

The v1-to-v2 migration preserves IDs, vector bytes, TTLs and version metadata.
Injected migration failures roll back the table rebuild and schema version, and
retry succeeds after the fault is removed. No automatic SQLite-to-PostgreSQL
migration exists. Such a migration would require a separately reviewed schema,
BLOB-to-BYTEA mapping, full metadata/TTL preservation, transactional imports,
row-count and vector checks, isolation tests and a rollback plan.

No scheduled backup, restore job, high availability or disaster recovery service
has been configured. A deployed service still needs filesystem ACLs/service-account
permissions, capacity monitoring, protected off-host backups and tested restores.
SQLite files contain plaintext approved query/response data. Namespaces remain
trusted scopes, not authenticated tenant boundaries.

## Dashboard and verification boundaries

System Information shows the configured backend and mode, configuration validity and
Not probed. Refresh does not connect to the database. An explicit verification
command is a separate observation and is not presented as ongoing dashboard health.
No database path, credentials or connection details are displayed.

The local dashboards read available JSONL history and saved benchmark reports.
The public Vercel dashboard serves packaged synthetic benchmark metrics and omits
local history. Its workspace API has returned production JSON. Hosted database
connectivity, Supabase migrations and production grants remain unverified.

Offline tests cover settings, initialization refusal, path encoding, connection
cleanup, timeout/lock recovery, read-only probes, incompatible schemas, migration
rollback/retry, persistence and namespace isolation. Existing cache tests remain
unchanged. Run:

    .\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider


## PostgreSQL adapter (Task 8A)

Select the backend on the server with `LCR_DATABASE_BACKEND=postgresql` and
`LCR_DATABASE_MODE=production`. Supply `LCR_POSTGRES_DSN` through the server's
secret manager. Use the provider-supplied Transaction Pooler hostname, port,
database and pooler username. Percent-encode credentials as URI components.
Only the URI options `sslmode=verify-full` and `sslrootcert` are accepted.
Install the trusted CA for libpq or supply its path with `sslrootcert`. Keep
the DSN out of browser variables, source files, command lines and logs.
Omitting the backend selects SQLite. Invalid PostgreSQL settings or failed
connections raise a sanitized error.

The driver pin is in `requirements.txt`. The server opens an autocommit Psycopg
3 connection per cache instance and releases it on context exit. Each database
operation has its own READ COMMITTED transaction, with a five-second connect
timeout, five-second statement timeout and two-second lock timeout. The adapter
enforces verified TLS and disables automatic prepared statements. Embedding and
Python similarity work occur outside database transactions.

PostgreSQL schema version 2 keeps the original query and the full stripped key.
A private-schema unique constraint indexes the 32-byte SHA-256 digest alongside
the compatibility scope. The adapter compares full keys whenever a digest
matches. A collision raises CacheConflictError, including when the existing
entry is expired. Ordinary semantic matches still use Python similarity and the
final invalidation and expiry recheck.

Every indexed text scope field has a **128 UTF-8 byte** budget in Python and SQL.
This includes namespace, embedding model/version/task/dtype, eligibility version,
cache version and serialized embedding configuration. The namespace bound also
protects the expiration index. The query key has no new length limit. PostgreSQL
uses its database clock for TTL creation, duplicate checks, lookup and cleanup.
A future-dated, unexpired winner blocks replacement. Both backends bind each
instance to one read-only namespace.

**Upgrade compatibility.** Preserve a backup and pause old writers before the
explicit upgrade. Migration 001 stays intact. Migration 002 verifies the v1
contract, takes an exclusive table lock, validates existing scopes, backfills
digests, installs the new arbiter and advances the version in one transaction.
Before cutover, check octet_length on all eight indexed text scope columns.
Oversized legacy scopes, a scoped digest collision, incompatible schema or a
timeout aborts the upgrade. Review conflicting rows separately; the migration
does not rename, truncate or discard keys. Older writers omit the now-required
digest, and older adapters expect version 1. For a large cache, rehearse the
backfill under the existing five-second statement and two-second lock budgets
on a disposable restored copy. Restore the backup to roll back after cutover.

Open and verify check UTF-8, table and column shape, types, nullability, identity
collations, validated checks, primary keys, expiration index and immediate B-tree
uniqueness suitable for the named conflict target. Resolve custom schema drift
before retrying. Namespaces remain caller-selected scopes.

**Deferred follow-up.** Lookup currently loads the full eligible scope into
memory; batching and its impact on deterministic winner selection need separate
work. Effective production role privileges and exposure of the private schema
also require a later authorized review. Offline catalog tests do not verify
Supabase grants, pooler behavior or PostgreSQL execution of the migration.

### Outstanding database verification steps

1. Install pinned dependencies in the project venv and run the offline suite:

       .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
       .\.venv\Scripts\python.exe -B -m pytest tests -q

2. Review the migration and privileges. For a disposable **local** engine check,
   create an empty database named `lcr_adapter_test` on IPv4 loopback with TLS
   and a trusted certificate valid for `127.0.0.1`. Provide its DSN in
   `LCR_TEST_LOCAL_PG_DSN` and set `LCR_TEST_LOCAL_PG=1`. Run
   `python -m pytest tests/test_postgres_local.py -q`, then unset the test
   variables and dispose of that database. The test refuses a preexisting
   application schema. Default runs skip this check.

3. For a subsequently authorized Supabase setup, confirm Free tier, no paid
   features or automatic billing, and the Transaction Pooler endpoint. Configure
   the CA and server-side DSN with a migration role. From the project root run
   `python database_admin.py initialize` once, then `python database_admin.py verify`.
   Initialization is explicit and repeatable for the recognized migration.
   These commands are future manual steps; Task 8A tests used synthetic data.

4. Switch the DSN to a dedicated runtime role. Grant CONNECT on the target
   database, USAGE on `lcr_cache`, SELECT on its version table and
   SELECT/INSERT/DELETE plus UPDATE(id) on its cache table (FOR UPDATE requires
   update privilege). Reserve schema creation for the migration role. Exclude
   the schema from exposed Data API schemas and check that public/anonymous roles
   have no access. Verify again as the runtime role. The probe validates schema
   access and indexes, not write permission. Any SQLite data transfer requires
   a separate reviewed backup, staging import, validation and rollback plan.

Psycopg references: [prepared statements](https://www.psycopg.org/psycopg3/docs/advanced/prepare.html)
and [transaction contexts](https://www.psycopg.org/psycopg3/docs/basic/transactions.html).
