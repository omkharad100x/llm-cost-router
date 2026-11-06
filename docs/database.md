# Database configuration — single-server MVP

## Decision and scope

Use Python's built-in SQLite on one application server with a durable local
volume. This deployment target was selected explicitly for this MVP. The cache
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

Managed PostgreSQL is the recommended next step if deployment needs multiple
hosts, significant concurrent writes or a platform without durable local disk.
It is not implemented here. A future adapter should use a supported PostgreSQL
driver, bounded pools, timeouts and certificate/hostname validation
([PostgreSQL TLS guidance](https://www.postgresql.org/docs/current/libpq-ssl.html)).
Free plans must be evaluated for storage, connection, compute, suspension and
backup limits at selection time. No provider, account or paid plan was created.

## Environment configuration

Only these process environment variables are read. No dotenv loading occurs.

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

System Information shows SQLite, the selected mode, configuration validity and
Not probed. Refresh does not connect to the database. An explicit verification
command is a separate observation and is not presented as ongoing dashboard health.
No database path, credentials or connection details are displayed.

The dashboards still read JSONL and saved benchmark reports from their existing
locations. A cache volume alone does not make those files persistent; deployment
must separately preserve them if needed. No application was deployed and no
production database is claimed operational.

Offline tests cover settings, initialization refusal, path encoding, connection
cleanup, timeout/lock recovery, read-only probes, incompatible schemas, migration
rollback/retry, persistence and namespace isolation. Existing cache tests remain
unchanged. Run:

    .\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
