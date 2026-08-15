//! CockroachDB-backed memory implementation.
//!
//! Ported from zeroclaw's `PostgresMemory` (`zeroclaw-memory/src/postgres.rs`,
//! MIT OR Apache-2.0 — see ATTRIBUTION.md) and adapted for CockroachDB:
//!
//! - Native `VECTOR` column type — no `CREATE EXTENSION vector`.
//! - `CREATE VECTOR INDEX (tenant_id, agent_id, embedding)` with prefix
//!   columns, so multi-tenant isolation lives in the index itself (C-SPANN).
//! - Hybrid recall: FTS keyword scoring (`ts_rank`; CockroachDB does not
//!   implement `ts_rank_cd`) merged with cosine-distance vector recall
//!   (`<=>`) via [`crate::vector::hybrid_merge`].
//! - Fresh final schema — the incremental Postgres migrations (and their
//!   `DO $$` blocks, which predate CockroachDB support) are not needed.
//! - Full `StoreOptions` persistence: kind / pinned / tenant_id are columns.
//!
//! TODO(kill-node demo): the sync `postgres` client holds one connection and
//! does not auto-reconnect. When a node dies mid-connection the next call
//! errors once, and a reconnect is needed. The demo server (ganglion-server)
//! will use an async pool with retry; this backend gets a reconnect wrapper
//! before the demo is recorded.

use crate::embeddings::{EmbeddingProvider, NoopEmbedding};
use crate::ledger::{
    self, ChainReport, LedgerEntry, SYSTEM_CHAIN, canonical_string, chain_for, content_sha256,
    hmac_hex, normalize_params, row_sha256,
};
use crate::traits::{
    Memory, MemoryCategory, MemoryEntry, MemoryKind, MemoryStats, StoreOptions,
    normalize_recent_recall_query, sanitize_session_key,
};
use crate::vector::hybrid_merge;
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, SecondsFormat, Timelike, Utc};
use parking_lot::Mutex;
use postgres::{Client, NoTls, Row, Transaction};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
use uuid::Uuid;

/// Maximum allowed connect timeout (seconds) to avoid unreasonable waits.
const CONNECT_TIMEOUT_CAP_SECS: u64 = 300;

/// Weights for hybrid recall fusion.
const VECTOR_WEIGHT: f32 = 0.6;
const KEYWORD_WEIGHT: f32 = 0.4;

struct DropOnThread<T: Send + 'static>(Option<T>);

impl<T: Send + 'static> DropOnThread<T> {
    fn new(value: T) -> Self {
        Self(Some(value))
    }
    fn get(&self) -> &T {
        self.0.as_ref().expect("DropOnThread value already taken")
    }
}

impl<T: Send + 'static> Drop for DropOnThread<T> {
    fn drop(&mut self) {
        let Some(value) = self.0.take() else { return };
        // postgres::Client::drop calls Runtime::block_on, which panics on a
        // Tokio runtime thread — hand the value to a plain OS thread. If the
        // OS refuses to spawn one, leak instead of panicking.
        let slot = std::mem::ManuallyDrop::new(value);
        if std::thread::Builder::new()
            .name("cockroach-client-drop".to_string())
            .spawn(move || drop(std::mem::ManuallyDrop::into_inner(slot)))
            .is_err()
        {
            tracing::warn!(
                "cockroach-client-drop thread spawn failed; leaking client to avoid nested-runtime panic"
            );
        }
    }
}

/// Everything needed to (re)establish a SQL connection. haproxy fronts the
/// cluster, so a reconnect after a node death lands on a surviving node —
/// this is the client half of the kill-node story.
struct ConnFactory {
    /// Already passed through [`sanitize_dsn`].
    sanitized_url: String,
    wants_tls: bool,
    connect_timeout: Option<Duration>,
}

impl ConnFactory {
    fn connect_once(&self) -> Result<Client> {
        let mut config: postgres::Config = self
            .sanitized_url
            .parse()
            .context("invalid CockroachDB connection URL")?;
        if let Some(t) = self.connect_timeout {
            config.connect_timeout(t);
        }
        if self.wants_tls {
            let connector = native_tls::TlsConnector::new().context("failed to build TLS connector")?;
            config
                .connect(postgres_native_tls::MakeTlsConnector::new(connector))
                .context("failed to connect to CockroachDB memory backend (TLS)")
        } else {
            config
                .connect(NoTls)
                .context("failed to connect to CockroachDB memory backend")
        }
    }

    /// Reconnect with backoff. Sized so a node death is survivable: haproxy's
    /// health check evicts a dead node within a few seconds; total budget here
    /// is ~15s across 8 attempts.
    fn connect_with_backoff(&self) -> Result<Client> {
        let mut last_err = None;
        for attempt in 0u32..8 {
            match self.connect_once() {
                Ok(c) => {
                    if attempt > 0 {
                        tracing::info!(attempt, "cockroach reconnect succeeded");
                    }
                    return Ok(c);
                }
                Err(e) => {
                    tracing::warn!(attempt, error = %e, "cockroach reconnect attempt failed");
                    last_err = Some(e);
                    std::thread::sleep(Duration::from_millis((100u64 << attempt).min(3_000)));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::Error::msg("reconnect failed")))
    }
}

/// A self-healing connection pool: N lazy [`Client`] slots plus the recipe to
/// rebuild any of them. [`SharedConn::run`] retries the caller's closure on a
/// fresh connection when the failure is connection-level (node died, socket
/// closed), and never for SQL-level errors.
///
/// Why a pool and not one mutexed connection: a single connection serializes
/// every operation, and a 40001 backoff sleeping inside the caller's closure
/// would hold that global lock — under a concurrent write load the whole
/// memory collapses to serial latency. With N slots, a slot sleeping through
/// its own retry backoff delays only itself. Slot count comes from
/// `GANGLION_POOL_SIZE` (default 8).
///
/// Retry semantics are at-least-once: a write whose ACK was lost may be
/// re-applied by the closure's own `with_txn_retry` structure on the fresh
/// connection. Belief writes tolerate this (an extra version, never a lost
/// one); the chaos suite asserts the stronger invariant that every ACKed
/// write is durable.
pub(crate) struct SharedConn {
    factory: ConnFactory,
    slots: Vec<Mutex<Option<Client>>>,
    next: std::sync::atomic::AtomicUsize,
}

pub(crate) fn pool_size_from_env() -> usize {
    std::env::var("GANGLION_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| (1..=64).contains(n))
        .unwrap_or(8)
}

impl SharedConn {
    /// Max reconnect *cycles* per operation (each cycle itself backs off up to
    /// 8 connection attempts inside [`ConnFactory::connect_with_backoff`]).
    const MAX_RECONNECT_CYCLES: u32 = 3;

    fn new(factory: ConnFactory, first: Client, size: usize) -> Self {
        let mut slots = Vec::with_capacity(size.max(1));
        slots.push(Mutex::new(Some(first)));
        for _ in 1..size.max(1) {
            slots.push(Mutex::new(None)); // connected lazily on first use
        }
        Self {
            factory,
            slots,
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn run<T>(&self, mut f: impl FnMut(&mut Client) -> Result<T>) -> Result<T> {
        // Round-robin slot pick; prefer an uncontended slot so one op sleeping
        // through a retry backoff doesn't queue unrelated ops behind it.
        let start = self
            .next
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let n = self.slots.len();
        let mut guard = 'pick: {
            for i in 0..n {
                if let Some(g) = self.slots[(start + i) % n].try_lock() {
                    break 'pick g;
                }
            }
            self.slots[start % n].lock()
        };

        let mut cycles = 0u32;
        loop {
            if guard.is_none() {
                *guard = Some(self.factory.connect_with_backoff()?);
            }
            let client = guard.as_mut().expect("connection just ensured");
            match f(client) {
                Err(e) if error_is_connection_level(&e) && cycles < Self::MAX_RECONNECT_CYCLES => {
                    cycles += 1;
                    tracing::warn!(cycle = cycles, error = %e, "connection-level failure; reconnecting");
                    // Safe to drop inline: `run` only executes on plain OS
                    // threads (every caller goes through run_on_os_thread),
                    // so Client::drop's block_on cannot hit a Tokio runtime.
                    drop(guard.take());
                }
                other => return other,
            }
        }
    }
}

/// Connection-level = the socket/session is gone (retry on a fresh connection
/// is sound); SQL-level errors (constraint violations, 40001, syntax) must
/// surface to the caller's own handling instead.
fn error_is_connection_level(e: &anyhow::Error) -> bool {
    let Some(pg) = e.downcast_ref::<postgres::Error>() else {
        return false;
    };
    if pg.is_closed() {
        return true;
    }
    if pg.as_db_error().is_some() {
        return false;
    }
    let mut source = std::error::Error::source(pg);
    while let Some(inner) = source {
        if inner.is::<std::io::Error>() {
            return true;
        }
        source = inner.source();
    }
    // rust-postgres wraps some transport failures in an opaque kind whose
    // io source is not exposed; the Display string is the documented tell.
    pg.to_string().contains("error communicating with the server")
}

pub struct CockroachMemory {
    client: DropOnThread<Arc<SharedConn>>,
    qualified_table: String,
    qualified_agents: String,
    qualified_ledger: String,
    qualified_edges: String,
    embedder: Arc<dyn EmbeddingProvider>,
    dimensions: usize,
    hmac_key: Arc<Vec<u8>>,
}

/// A knowledge-graph edge attached to a belief write, committed in the same
/// transaction as the belief row itself.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edge {
    /// Target memory row id. Enforced by a foreign key: a dangling target
    /// aborts (and rolls back) the whole hybrid write.
    pub to_id: String,
    pub relation: String,
}

/// Result of a belief write ([`CockroachMemory::store_belief`] /
/// [`CockroachMemory::supersede_belief`]).
#[derive(Debug, Clone, serde::Serialize)]
pub struct BeliefWritten {
    /// Row id of the now-current belief version.
    pub id: String,
    /// Row id of the version this write closed, if any.
    pub superseded_id: Option<String>,
    /// Ledger entry appended in the same transaction.
    pub ledger_id: i64,
}

/// One scope chain's verification result.
#[derive(Debug, serde::Serialize)]
pub struct ChainVerification {
    pub chain_id: String,
    #[serde(flatten)]
    pub report: ChainReport,
}

/// Full ledger verification: per-chain HMAC verification plus row-state
/// cross-check.
#[derive(Debug, serde::Serialize)]
pub struct LedgerVerification {
    pub chains: Vec<ChainVerification>,
    /// Memory rows whose current state hash disagrees with the latest ledger
    /// commitment for that row — direct SQL tampering.
    pub row_mismatches: Vec<RowMismatch>,
    pub rows_checked: usize,
}

impl LedgerVerification {
    pub fn is_clean(&self) -> bool {
        self.chains.iter().all(|c| c.report.valid) && self.row_mismatches.is_empty()
    }

    pub fn entries_checked(&self) -> usize {
        self.chains.iter().map(|c| c.report.checked).sum()
    }
}

#[derive(Debug, serde::Serialize)]
pub struct RowMismatch {
    pub id: String,
    pub key: String,
    pub ledger_sha256: String,
    pub row_sha256: String,
}

impl CockroachMemory {
    /// Connect and initialize the schema. `embedder` with dimensions > 0
    /// enables vector storage/recall; pass [`NoopEmbedding`] for keyword-only.
    pub fn new(
        db_url: &str,
        schema: &str,
        table: &str,
        connect_timeout_secs: Option<u64>,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
    ) -> Result<Self> {
        validate_identifier(schema, "storage schema")?;
        validate_identifier(table, "storage table")?;

        let embedder: Arc<dyn EmbeddingProvider> =
            embedder.unwrap_or_else(|| Arc::new(NoopEmbedding));
        let dimensions = embedder.dimensions();

        let schema_ident = quote_identifier(schema);
        let table_ident = quote_identifier(table);
        let qualified_table = format!("{schema_ident}.{table_ident}");
        let qualified_agents = format!("{schema_ident}.agents");
        let qualified_ledger = format!("{schema_ident}.memory_ledger");
        let qualified_edges = format!("{schema_ident}.memory_edges");

        let client = Self::initialize_client(
            db_url.to_string(),
            connect_timeout_secs,
            schema_ident,
            qualified_table.clone(),
            qualified_agents.clone(),
            qualified_ledger.clone(),
            qualified_edges.clone(),
            dimensions,
        )?;

        Ok(Self {
            client: DropOnThread::new(Arc::new(client)),
            qualified_table,
            qualified_agents,
            qualified_ledger,
            qualified_edges,
            embedder,
            dimensions,
            hmac_key: Arc::new(ledger::key_from_env()),
        })
    }

    /// Replace the ledger signing key (constructor reads `GANGLION_HMAC_KEY`).
    pub fn with_hmac_key(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.hmac_key = Arc::new(key.into());
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn initialize_client(
        db_url: String,
        connect_timeout_secs: Option<u64>,
        schema_ident: String,
        qualified_table: String,
        qualified_agents: String,
        qualified_ledger: String,
        qualified_edges: String,
        dimensions: usize,
    ) -> Result<SharedConn> {
        let init_handle = std::thread::Builder::new()
            .name("cockroach-memory-init".to_string())
            .spawn(move || -> Result<SharedConn> {
                // rust-postgres only parses sslmode=disable|prefer|require.
                // CockroachDB Cloud hands out verify-full: map it for the
                // parser; native-tls still verifies the server cert against
                // system roots (cockroachlabs.cloud uses a public CA).
                // CockroachDB Cloud requires TLS; local --insecure clusters
                // don't speak it. Pick by the DSN's sslmode.
                let factory = ConnFactory {
                    sanitized_url: sanitize_dsn(&db_url),
                    wants_tls: ["sslmode=require", "sslmode=verify-ca", "sslmode=verify-full"]
                        .iter()
                        .any(|m| db_url.contains(m)),
                    connect_timeout: connect_timeout_secs
                        .map(|s| Duration::from_secs(s.min(CONNECT_TIMEOUT_CAP_SECS))),
                };

                let mut client = factory.connect_once()?;
                Self::init_schema(
                    &mut client,
                    &schema_ident,
                    &qualified_table,
                    &qualified_agents,
                    &qualified_ledger,
                    &qualified_edges,
                    dimensions,
                )?;
                Ok(SharedConn::new(factory, client, pool_size_from_env()))
            })
            .context("failed to spawn CockroachDB initializer thread")?;

        init_handle.join().map_err(|_| {
            tracing::error!("CockroachDB initializer thread panicked");
            anyhow::Error::msg("CockroachDB initializer thread panicked")
        })?
    }

    #[allow(clippy::too_many_arguments)]
    fn init_schema(
        client: &mut Client,
        schema_ident: &str,
        qualified_table: &str,
        qualified_agents: &str,
        qualified_ledger: &str,
        qualified_edges: &str,
        dimensions: usize,
    ) -> Result<()> {
        // Concurrent schema creation contends on shared descriptor tables and
        // legitimately yields 40001 — every init statement runs under retry.
        let ddl_agents = format!(
            "
            CREATE SCHEMA IF NOT EXISTS {schema_ident};

            CREATE TABLE IF NOT EXISTS {qualified_agents} (
                id          TEXT PRIMARY KEY,
                alias       TEXT NOT NULL UNIQUE,
                created_at  TIMESTAMPTZ NOT NULL
            );
            "
        );
        with_txn_retry(|| client.batch_execute(&ddl_agents))?;

        // Seed the default agent (attribution target when none is given).
        let candidate = Uuid::new_v4().to_string();
        let seed = format!(
            "INSERT INTO {qualified_agents} (id, alias, created_at)
             VALUES ($1, 'default', NOW())
             ON CONFLICT (alias) DO NOTHING"
        );
        with_txn_retry(|| client.execute(&seed, &[&candidate]))?;

        let embedding_column = if dimensions > 0 {
            format!(",\n                embedding VECTOR({dimensions})")
        } else {
            String::new()
        };

        // Bitemporal belief rows: a key's history is a chain of rows sharing
        // (agent_id, key); exactly one is *current* (valid_to IS NULL) —
        // enforced by the partial unique index (verified: CockroachDB v26.2
        // supports both the partial index and its use as ON CONFLICT arbiter).
        let ddl_memories = format!(
            "
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                id            TEXT PRIMARY KEY,
                tenant_id     TEXT NOT NULL DEFAULT 'default',
                agent_id      TEXT NOT NULL REFERENCES {qualified_agents}(id),
                namespace     TEXT NOT NULL DEFAULT 'default',
                key           TEXT NOT NULL,
                content       TEXT NOT NULL,
                category      TEXT NOT NULL,
                kind          TEXT,
                importance    FLOAT8,
                pinned        BOOL NOT NULL DEFAULT false,
                superseded_by TEXT,
                session_id    TEXT,
                source        TEXT,
                valid_from    TIMESTAMPTZ NOT NULL,
                valid_to      TIMESTAMPTZ,
                created_at    TIMESTAMPTZ NOT NULL,
                updated_at    TIMESTAMPTZ NOT NULL{embedding_column}
            );

            CREATE UNIQUE INDEX IF NOT EXISTS memories_current_uniq
                ON {qualified_table} (agent_id, key) WHERE valid_to IS NULL;
            CREATE INDEX IF NOT EXISTS idx_memories_category ON {qualified_table}(category);
            CREATE INDEX IF NOT EXISTS idx_memories_session_id ON {qualified_table}(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON {qualified_table}(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON {qualified_table}(namespace);
            CREATE INDEX IF NOT EXISTS idx_memories_validity ON {qualified_table}(key, valid_from, valid_to);
            CREATE INDEX IF NOT EXISTS idx_memories_content_fts ON {qualified_table} USING gin(to_tsvector('simple', content));
            CREATE INDEX IF NOT EXISTS idx_memories_key_fts ON {qualified_table} USING gin(to_tsvector('simple', translate(key, '_-', '  ')));
            "
        );
        with_txn_retry(|| client.batch_execute(&ddl_memories))?;

        // Tamper-evident ledger + knowledge-graph edges. No advisory locks on
        // CockroachDB: the head read + PK insert under SERIALIZABLE serializes
        // appends (concurrent appends conflict, one retries via 40001).
        // Edge FKs cascade so `forget` can delete versioned rows.
        let ddl_ledger = format!(
            "
            CREATE TABLE IF NOT EXISTS {qualified_ledger} (
                chain_id   TEXT NOT NULL,
                id         INT8 NOT NULL,
                ts         TIMESTAMPTZ NOT NULL,
                kind       TEXT NOT NULL,
                actor      TEXT NOT NULL,
                target     TEXT NOT NULL,
                params     JSONB NOT NULL,
                prev_hmac  TEXT NOT NULL,
                hmac       TEXT NOT NULL,
                PRIMARY KEY (chain_id, id)
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_target ON {qualified_ledger}(target, ts DESC);

            CREATE TABLE IF NOT EXISTS {qualified_edges} (
                from_id    TEXT NOT NULL REFERENCES {qualified_table}(id) ON DELETE CASCADE,
                to_id      TEXT NOT NULL REFERENCES {qualified_table}(id) ON DELETE CASCADE,
                relation   TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (from_id, relation, to_id)
            );
            "
        );
        with_txn_retry(|| client.batch_execute(&ddl_ledger))?;

        if dimensions > 0 {
            // The embedding column's dimensionality is frozen at first boot.
            // Booting later with a different provider (HashEmbedding(32) →
            // fastembed(384)) would otherwise fail on every single write with
            // an opaque cast error — check once at init and say it plainly.
            let col: Option<String> = with_txn_retry(|| {
                Ok(client
                    .query(
                        &format!(
                            "SELECT data_type FROM [SHOW COLUMNS FROM {qualified_table}]
                             WHERE column_name = 'embedding'"
                        ),
                        &[],
                    )?
                    .first()
                    .map(|r| r.get(0)))
            })?;
            match col.as_deref() {
                None => anyhow::bail!(
                    "table {qualified_table} exists without an `embedding` column, but the \
                     configured embedding provider needs VECTOR({dimensions}); migrate the \
                     table or use a fresh schema"
                ),
                Some(t) => {
                    let existing: Option<usize> = t
                        .strip_prefix("VECTOR(")
                        .and_then(|s| s.strip_suffix(')'))
                        .and_then(|s| s.parse().ok());
                    if existing != Some(dimensions) {
                        anyhow::bail!(
                            "embedding column of {qualified_table} is {t}, but the configured \
                             embedding provider produces {dimensions} dimensions; drop/migrate \
                             the table or match the provider to the stored dimensionality"
                        );
                    }
                }
            }

            // C-SPANN distributed vector index. The (tenant_id, agent_id)
            // prefix columns partition the ANN search space per tenant/agent.
            let ddl_vec = format!(
                "CREATE VECTOR INDEX IF NOT EXISTS idx_memories_embedding
                 ON {qualified_table} (tenant_id, agent_id, embedding)"
            );
            with_txn_retry(|| client.batch_execute(&ddl_vec))?;
        }

        Ok(())
    }

    fn category_to_str(category: &MemoryCategory) -> String {
        category.to_string()
    }

    fn parse_category(value: &str) -> MemoryCategory {
        match value {
            "core" => MemoryCategory::Core,
            "daily" => MemoryCategory::Daily,
            "conversation" => MemoryCategory::Conversation,
            other => MemoryCategory::Custom(other.to_string()),
        }
    }

    /// Text form CockroachDB accepts for a `::vector` cast: `[x,y,...]`.
    fn vector_literal(v: &[f32]) -> String {
        let mut s = String::with_capacity(v.len() * 10 + 2);
        s.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("{x}"));
        }
        s.push(']');
        s
    }

    fn row_to_entry(row: &Row) -> Result<MemoryEntry> {
        // Named access so this is immune to SELECT column reordering.
        let timestamp: DateTime<Utc> = row.get("created_at");
        let kind: Option<MemoryKind> = row
            .try_get::<_, Option<String>>("kind")
            .ok()
            .flatten()
            .and_then(|raw| serde_json::from_str(&raw).ok());

        Ok(MemoryEntry {
            id: row.get("id"),
            key: row.get("key"),
            content: row.get("content"),
            category: Self::parse_category(&row.get::<_, String>("category")),
            timestamp: timestamp.to_rfc3339(),
            session_id: row.get("session_id"),
            score: row.try_get("score").ok(),
            namespace: row
                .try_get::<_, String>("namespace")
                .unwrap_or_else(|_| "default".into()),
            importance: row.try_get("importance").ok().flatten(),
            superseded_by: row.try_get("superseded_by").ok().flatten(),
            kind,
            pinned: row.try_get("pinned").unwrap_or(false),
            tenant_id: row.try_get("tenant_id").ok(),
            agent_alias: row.try_get("agent_alias").ok(),
            agent_id: row.try_get("agent_id").ok(),
        })
    }

    /// Columns every entry-producing SELECT must expose (plus optional `score`).
    fn entry_columns() -> &'static str {
        "m.id, m.key, m.content, m.category, m.created_at, m.session_id,
         m.namespace, m.importance, m.superseded_by, m.kind, m.pinned,
         m.tenant_id, a.alias AS agent_alias, m.agent_id"
    }

    async fn embed_query(&self, query: &str) -> Option<Vec<f32>> {
        if self.dimensions == 0 || query.is_empty() {
            return None;
        }
        match self.embedder.embed_one(query).await {
            Ok(v) if v.len() == self.dimensions => Some(v),
            Ok(_) => {
                tracing::warn!("query embedding has wrong dimensionality; skipping vector recall");
                None
            }
            Err(err) => {
                tracing::warn!("query embedding failed ({err}); keyword-only recall");
                None
            }
        }
    }

    /// Shared hybrid recall over an optional agent allowlist.
    #[allow(clippy::too_many_arguments)]
    async fn recall_hybrid(
        &self,
        allowed_agent_ids: Option<Vec<String>>,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let query = normalize_recent_recall_query(query).trim().to_string();
        let query_vec = self.embed_query(&query).await;
        // Sanitized on write; sanitize on read or the filter never matches.
        let sid = session_id.map(sanitize_session_key);
        let since_owned = since.map(str::to_string);
        let until_owned = until.map(str::to_string);

        run_on_os_thread(move || -> Result<Vec<MemoryEntry>> {
            client.run(|client| {
            let columns = Self::entry_columns();
            #[allow(clippy::cast_possible_wrap)]
            let limit_i64 = limit as i64;
            let allowed_ref: Option<&Vec<String>> = allowed_agent_ids.as_ref();

            // Fixed params $1..$4; time bounds appended after. The TEXT
            // double-casts keep the wire parameter types TEXT — rust-postgres
            // cannot serialize String into timestamptz (same story as vector).
            let agent_filter = " AND ($4::TEXT[] IS NULL OR m.agent_id = ANY($4))";
            let time_filter: String = match (since_owned.as_deref(), until_owned.as_deref()) {
                (Some(_), Some(_)) => {
                    " AND m.created_at >= $5::TEXT::TIMESTAMPTZ AND m.created_at <= $6::TEXT::TIMESTAMPTZ"
                        .into()
                }
                (Some(_), None) => " AND m.created_at >= $5::TEXT::TIMESTAMPTZ".into(),
                (None, Some(_)) => " AND m.created_at <= $5::TEXT::TIMESTAMPTZ".into(),
                (None, None) => String::new(),
            };

            // Empty query = recency-only recall. This is a dedicated statement
            // because CockroachDB (unlike Postgres) errors on
            // plainto_tsquery('simple', '') — "doesn't contain lexemes" —
            // even inside a short-circuitable CASE/OR.
            // Keyword arms run plainto_tsquery('simple', $1), which
            // ERRORS on a lexeme-free query (punctuation-only) instead of
            // matching nothing — so a lexeme-free query must never reach
            // them. Vector recall still works for such queries.
            let has_lexemes = query.chars().any(char::is_alphanumeric);
            if !has_lexemes && query_vec.is_none() {
                let recency_stmt = format!(
                    "
                    SELECT {columns}, 0.0::FLOAT8 AS score
                    FROM {qualified_table} m
                    LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                    WHERE ($2::TEXT IS NULL OR m.session_id = $2)
                      AND ($1 = '' OR true)
                      AND m.valid_to IS NULL
                      {agent_filter}
                      {time_filter}
                    ORDER BY m.updated_at DESC
                    LIMIT $3
                    ",
                );
                let mut params: Vec<&(dyn postgres::types::ToSql + Sync)> =
                    vec![&query, &sid, &limit_i64, &allowed_ref];
                match (since_owned.as_deref(), until_owned.as_deref()) {
                    (Some(_), Some(_)) => {
                        params.push(&since_owned);
                        params.push(&until_owned);
                    }
                    (Some(_), None) => params.push(&since_owned),
                    (None, Some(_)) => params.push(&until_owned),
                    (None, None) => {}
                }
                let rows = client.query(&recency_stmt, &params)?;
                return rows
                    .iter()
                    .map(Self::row_to_entry)
                    .collect::<Result<Vec<MemoryEntry>>>();
            }

            // Keys are frequently snake_case/kebab-case; translate() splits
            // them into lexemes so "deploy runbook" matches key deploy_runbook.
            let keyword_stmt = format!(
                "
                SELECT {columns},
                       (
                         CASE WHEN to_tsvector('simple', translate(m.key, '_-', '  ')) @@ plainto_tsquery('simple', $1)
                           THEN ts_rank(to_tsvector('simple', translate(m.key, '_-', '  ')), plainto_tsquery('simple', $1)) * 2.0
                           ELSE 0.0 END +
                         CASE WHEN to_tsvector('simple', m.content) @@ plainto_tsquery('simple', $1)
                           THEN ts_rank(to_tsvector('simple', m.content), plainto_tsquery('simple', $1))
                           ELSE 0.0 END
                       )::FLOAT8 AS score
                FROM {qualified_table} m
                LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                WHERE ($2::TEXT IS NULL OR m.session_id = $2)
                  AND to_tsvector('simple', translate(m.key, '_-', '  ') || ' ' || m.content) @@ plainto_tsquery('simple', $1)
                  AND m.valid_to IS NULL
                  {agent_filter}
                  {time_filter}
                ORDER BY score DESC, m.updated_at DESC
                LIMIT $3
                ",
            );

            let keyword_rows: Vec<Row> = if !has_lexemes {
                Vec::new()
            } else {
                let mut params: Vec<&(dyn postgres::types::ToSql + Sync)> =
                    vec![&query, &sid, &limit_i64, &allowed_ref];
                match (since_owned.as_deref(), until_owned.as_deref()) {
                    (Some(_), Some(_)) => {
                        params.push(&since_owned);
                        params.push(&until_owned);
                    }
                    (Some(_), None) => params.push(&since_owned),
                    (None, Some(_)) => params.push(&until_owned),
                    (None, None) => {}
                }
                client.query(&keyword_stmt, &params)?
            };

            // Vector arm: nearest neighbors by cosine distance.
            let vector_rows: Vec<Row> = if let Some(qv) = query_vec.as_ref() {
                let qlit = Self::vector_literal(qv);
                let vector_stmt = format!(
                    "
                    SELECT {columns},
                           (1.0 - (m.embedding <=> $1::TEXT::vector))::FLOAT8 AS score
                    FROM {qualified_table} m
                    LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                    WHERE m.embedding IS NOT NULL
                      AND ($2::TEXT IS NULL OR m.session_id = $2)
                      AND m.valid_to IS NULL
                      {agent_filter}
                      {time_filter}
                    ORDER BY m.embedding <=> $1::TEXT::vector
                    LIMIT $3
                    ",
                );
                let mut params: Vec<&(dyn postgres::types::ToSql + Sync)> =
                    vec![&qlit, &sid, &limit_i64, &allowed_ref];
                match (since_owned.as_deref(), until_owned.as_deref()) {
                    (Some(_), Some(_)) => {
                        params.push(&since_owned);
                        params.push(&until_owned);
                    }
                    (Some(_), None) => params.push(&since_owned),
                    (None, Some(_)) => params.push(&until_owned),
                    (None, None) => {}
                }
                client.query(&vector_stmt, &params)?
            } else {
                Vec::new()
            };

            // Recency path (empty query, no vector): keyword_stmt already
            // returned recency-ordered rows with score 0 — pass through.
            if !has_lexemes && vector_rows.is_empty() {
                return keyword_rows
                    .iter()
                    .map(Self::row_to_entry)
                    .collect::<Result<Vec<MemoryEntry>>>();
            }

            // Hybrid fusion by id.
            let mut entries: HashMap<String, MemoryEntry> = HashMap::new();
            let mut keyword_scores: Vec<(String, f32)> = Vec::new();
            for row in &keyword_rows {
                let entry = Self::row_to_entry(row)?;
                #[allow(clippy::cast_possible_truncation)]
                let s = entry.score.unwrap_or(0.0) as f32;
                if s > 0.0 {
                    keyword_scores.push((entry.id.clone(), s));
                }
                entries.insert(entry.id.clone(), entry);
            }
            let mut vector_scores: Vec<(String, f32)> = Vec::new();
            for row in &vector_rows {
                let entry = Self::row_to_entry(row)?;
                #[allow(clippy::cast_possible_truncation)]
                let s = entry.score.unwrap_or(0.0) as f32;
                vector_scores.push((entry.id.clone(), s));
                entries.entry(entry.id.clone()).or_insert(entry);
            }

            let merged = hybrid_merge(
                &vector_scores,
                &keyword_scores,
                VECTOR_WEIGHT,
                KEYWORD_WEIGHT,
                limit,
            );

            Ok(merged
                .into_iter()
                .filter_map(|scored| {
                    entries.remove(&scored.id).map(|mut e| {
                        e.score = Some(f64::from(scored.final_score));
                        e
                    })
                })
                .collect())
            })
        })
        .await
    }
}

async fn run_on_os_thread<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = oneshot::channel();

    std::thread::Builder::new()
        .name("cockroach-memory-op".to_string())
        .spawn(move || {
            let result = f();
            let _ = tx.send(result);
        })
        .context("failed to spawn CockroachDB operation thread")?;

    rx.await.map_err(|_| {
        tracing::error!("CockroachDB operation thread terminated unexpectedly");
        anyhow::Error::msg("CockroachDB operation thread terminated unexpectedly")
    })?
}

/// CockroachDB's serializable isolation surfaces retryable conflicts as
/// SQLSTATE 40001 and expects the client to retry the transaction — this is
/// the documented contract, not an error condition.
/// <https://www.cockroachlabs.com/docs/stable/transaction-retry-error-reference>
fn is_retryable(e: &postgres::Error) -> bool {
    e.code() == Some(&postgres::error::SqlState::T_R_SERIALIZATION_FAILURE)
}

/// Run a statement (or whole transaction closure) with 40001 retry + backoff.
fn with_txn_retry<T, F>(mut f: F) -> std::result::Result<T, postgres::Error>
where
    F: FnMut() -> std::result::Result<T, postgres::Error>,
{
    let mut attempt: u32 = 0;
    loop {
        match f() {
            Err(e) if is_retryable(&e) && attempt < 7 => {
                attempt += 1;
                std::thread::sleep(Duration::from_millis(15u64 << attempt));
            }
            other => return other,
        }
    }
}

/// Truncate to microseconds: timestamptz stores microseconds and the ledger's
/// canonical string must survive a round-trip through the database.
fn now_micros() -> DateTime<Utc> {
    let now = Utc::now();
    now.with_nanosecond((now.nanosecond() / 1000) * 1000)
        .unwrap_or(now)
}

fn ts_micros_string(ts: &DateTime<Utc>) -> String {
    ts.to_rfc3339_opts(SecondsFormat::Micros, true)
}

/// Append one ledger entry inside the caller's transaction, on the given
/// scope chain.
///
/// Per-chain head read + `id = head + 1` PK insert under SERIALIZABLE
/// isolation is the append serializer (CockroachDB has no advisory locks):
/// two concurrent appends *to the same chain* read the same head, one
/// commits, the other gets 40001 and the caller's `with_txn_retry` re-runs
/// the whole transaction — mutation and ledger entry stay atomic across
/// retries by construction. Appends to different chains don't contend at
/// all, which is the point: a single global chain is a hot key that starves
/// concurrent writers into retry exhaustion.
#[allow(clippy::too_many_arguments)]
fn ledger_append_in_tx(
    tx: &mut Transaction<'_>,
    qualified_ledger: &str,
    hmac_key: &[u8],
    chain_id: &str,
    kind: &str,
    actor: &str,
    target: &str,
    params: serde_json::Value,
) -> std::result::Result<i64, postgres::Error> {
    let head = tx.query_opt(
        &format!(
            "SELECT id, hmac FROM {qualified_ledger} WHERE chain_id = $1 ORDER BY id DESC LIMIT 1"
        ),
        &[&chain_id],
    )?;
    let (prev_id, prev_hmac) = match head {
        Some(row) => (row.get::<_, i64>(0), row.get::<_, String>(1)),
        None => (0, ledger::GENESIS.to_string()),
    };
    let id = prev_id + 1;
    let ts_dt = now_micros();
    let ts = ts_micros_string(&ts_dt);
    let params = normalize_params(&params);
    let canonical = canonical_string(chain_id, id, &ts, kind, actor, target, &params, &prev_hmac);
    let hmac = hmac_hex(hmac_key, &canonical);
    tx.execute(
        &format!(
            "INSERT INTO {qualified_ledger} (chain_id, id, ts, kind, actor, target, params, prev_hmac, hmac)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)"
        ),
        &[
            &chain_id, &id, &ts_dt, &kind, &actor, &target, &params, &prev_hmac, &hmac,
        ],
    )?;
    Ok(id)
}

/// Make a CockroachDB Cloud DSN palatable to rust-postgres: map
/// verify-full/verify-ca to require (TLS verification still happens in
/// native-tls against system roots) and drop options the parser rejects
/// (`sslrootcert`, ccloud's cluster routing helpers).
fn sanitize_dsn(db_url: &str) -> String {
    let Some((base, query)) = db_url.split_once('?') else {
        return db_url.to_string();
    };
    let kept: Vec<String> = query
        .split('&')
        .filter(|seg| !seg.starts_with("sslrootcert="))
        .map(|seg| match seg {
            "sslmode=verify-full" | "sslmode=verify-ca" => "sslmode=require".to_string(),
            other => other.to_string(),
        })
        .collect();
    if kept.is_empty() {
        base.to_string()
    } else {
        format!("{base}?{}", kept.join("&"))
    }
}

fn validate_identifier(value: &str, field_name: &str) -> Result<()> {
    if value.is_empty() {
        anyhow::bail!("{field_name} must not be empty");
    }

    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        anyhow::bail!("{field_name} must not be empty");
    };

    if !(first.is_ascii_alphabetic() || first == '_') {
        anyhow::bail!("{field_name} must start with an ASCII letter or underscore; got '{value}'");
    }

    if !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_') {
        anyhow::bail!(
            "{field_name} can only contain ASCII letters, numbers, and underscores; got '{value}'"
        );
    }

    Ok(())
}

fn quote_identifier(value: &str) -> String {
    format!("\"{value}\"")
}

/// One version in a belief's history.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BeliefVersion {
    pub id: String,
    pub key: String,
    pub content: String,
    pub source: Option<String>,
    /// RFC 3339. When this version became the current belief.
    pub valid_from: String,
    /// RFC 3339. When it stopped being current; `None` = still current.
    pub valid_to: Option<String>,
    pub superseded_by: Option<String>,
}

impl CockroachMemory {
    /// Versioned "remember": insert the belief if the key has no current
    /// version, otherwise supersede the current version — never overwrite.
    ///
    /// One SERIALIZABLE transaction commits: the new row (with embedding),
    /// the closing of the previous version, every knowledge-graph edge, and
    /// the ledger entry. A dangling edge target aborts all of it.
    pub async fn store_belief(
        &self,
        key: &str,
        content: &str,
        source: Option<&str>,
        agent_id: Option<&str>,
        options: StoreOptions,
        edges: &[Edge],
    ) -> Result<BeliefWritten> {
        self.write_belief(key, content, source, agent_id, options, edges, false)
            .await
    }

    /// Correct an existing belief: close the current version and open the new
    /// one in the same transaction. Errors if the key has no current version
    /// (you cannot correct a belief that was never held).
    pub async fn supersede_belief(
        &self,
        key: &str,
        content: &str,
        source: Option<&str>,
        agent_id: Option<&str>,
        options: StoreOptions,
        edges: &[Edge],
    ) -> Result<BeliefWritten> {
        self.write_belief(key, content, source, agent_id, options, edges, true)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn write_belief(
        &self,
        key: &str,
        content: &str,
        source: Option<&str>,
        agent_id: Option<&str>,
        options: StoreOptions,
        edges: &[Edge],
        require_existing: bool,
    ) -> Result<BeliefWritten> {
        let embedding = if self.dimensions > 0 {
            let embed_text = format!("{key}\n{content}");
            match self.embedder.embed_one(&embed_text).await {
                Ok(v) if v.len() == self.dimensions => Some(Self::vector_literal(&v)),
                Ok(_) => {
                    tracing::warn!("belief embedding has wrong dimensionality; storing without");
                    None
                }
                Err(err) => {
                    tracing::warn!("belief embedding failed ({err}); storing without");
                    None
                }
            }
        } else {
            None
        };

        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let qualified_edges = self.qualified_edges.clone();
        let hmac_key = self.hmac_key.clone();
        let key = key.to_string();
        let content = content.to_string();
        let source = source.map(str::to_string);
        let aid = agent_id.map(str::to_string);
        let namespace = options.namespace.unwrap_or_else(|| "default".into());
        let importance = options.importance;
        let kind_json = options
            .kind
            .as_ref()
            .map(|k| serde_json::to_string(k))
            .transpose()?;
        let pinned = options.pinned;
        let tenant_id = options.tenant_id.unwrap_or_else(|| "default".into());
        let edges: Vec<Edge> = edges.to_vec();
        let has_vector = self.dimensions > 0;

        run_on_os_thread(move || -> Result<BeliefWritten> {
            client.run(|client| {
            let content_hash = content_sha256(&content);

            let select_current = format!(
                "SELECT id, content, valid_from, importance, pinned FROM {qualified_table}
                 WHERE agent_id = $1 AND key = $2 AND valid_to IS NULL
                 FOR UPDATE"
            );
            let close_old = format!(
                "UPDATE {qualified_table}
                 SET valid_to = $1::TEXT::TIMESTAMPTZ, superseded_by = $2,
                     updated_at = $1::TEXT::TIMESTAMPTZ
                 WHERE id = $3"
            );
            let (embedding_col, embedding_val) = if has_vector {
                (", embedding", ", $13::TEXT::vector")
            } else {
                ("", "")
            };
            let insert_new = format!(
                "INSERT INTO {qualified_table}
                    (id, tenant_id, agent_id, namespace, key, content, category,
                     kind, importance, pinned, session_id, source,
                     created_at, updated_at, valid_from{embedding_col})
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NULL, $11,
                         $12::TEXT::TIMESTAMPTZ, $12::TEXT::TIMESTAMPTZ,
                         $12::TEXT::TIMESTAMPTZ{embedding_val})"
            );
            let insert_edge = format!(
                "INSERT INTO {qualified_edges} (from_id, to_id, relation, created_at)
                 VALUES ($1, $2, $3, NOW())"
            );

            let written = with_txn_retry(|| {
                let mut tx = client.transaction()?;

                let agent_uuid: String = match aid.as_deref() {
                    Some(a) => a.to_string(),
                    None => tx
                        .query_one(
                            &format!(
                                "SELECT id FROM {qualified_agents} WHERE alias = 'default' LIMIT 1"
                            ),
                            &[],
                        )?
                        .get(0),
                };

                // Old-row fields feed the closed row's post-supersede hash:
                // the ledger must commit to the old version's final state
                // (valid_to = now) or a later valid_to flip would be invisible.
                let old_row: Option<(String, String, DateTime<Utc>, Option<f64>, bool)> = tx
                    .query_opt(&select_current, &[&agent_uuid, &key])?
                    .map(|r| (r.get(0), r.get(1), r.get(2), r.get(3), r.get(4)));

                // Strict supersede: no current belief → nothing to correct.
                // Abort before any write; the dropped tx rolls back.
                if require_existing && old_row.is_none() {
                    return Ok(None);
                }

                let new_id = Uuid::new_v4().to_string();
                // TEXT-cast timestamps (same wire-type story as the vector
                // casts); micros so the string round-trips through the DB.
                let now_s = ts_micros_string(&now_micros());

                let old_id: Option<String> = old_row.as_ref().map(|(id, ..)| id.clone());
                let superseded_row_sha = old_row.as_ref().map(|(_, c, vf, imp, pin)| {
                    row_sha256(&key, c, &ts_micros_string(vf), Some(&now_s), *imp, *pin)
                });
                if let Some(old) = old_id.as_deref() {
                    tx.execute(&close_old, &[&now_s, &new_id, &old])?;
                }

                let mut params: Vec<&(dyn postgres::types::ToSql + Sync)> = vec![
                    &new_id,
                    &tenant_id,
                    &agent_uuid,
                    &namespace,
                    &key,
                    &content,
                    &"core",
                    &kind_json,
                    &importance,
                    &pinned,
                    &source,
                    &now_s,
                ];
                if has_vector {
                    params.push(&embedding);
                }
                tx.execute(&insert_new, &params)?;

                for e in &edges {
                    tx.execute(&insert_edge, &[&new_id, &e.to_id, &e.relation])?;
                }

                let ledger_kind = if old_id.is_some() { "supersede" } else { "store" };
                let new_row_sha = row_sha256(&key, &content, &now_s, None, importance, pinned);
                let ledger_id = ledger_append_in_tx(
                    &mut tx,
                    &qualified_ledger,
                    &hmac_key,
                    &chain_for(&tenant_id, &agent_uuid),
                    ledger_kind,
                    &agent_uuid,
                    &new_id,
                    json!({
                        "key": key,
                        "content_sha256": content_hash,
                        "row_sha256": new_row_sha,
                        "supersedes": old_id,
                        "superseded_row_sha256": superseded_row_sha,
                        "source": source,
                        "edges": edges.len(),
                    }),
                )?;

                tx.commit()?;
                Ok(Some((new_id, old_id, ledger_id)))
            });

            let Some((new_id, old_id, ledger_id)) = written? else {
                anyhow::bail!(
                    "supersede on key '{key}' which had no current belief; use store_belief for first assertions"
                );
            };
            Ok(BeliefWritten {
                id: new_id,
                superseded_id: old_id,
                ledger_id,
            })
            })
        })
        .await
    }

    /// Full version history of a key **for one agent**, oldest first. Two
    /// agents legitimately hold the same key; an unscoped timeline would
    /// interleave their chains. `agent_id = None` resolves to the default
    /// agent, mirroring the write path.
    pub async fn belief_timeline(
        &self,
        key: &str,
        agent_id: Option<&str>,
    ) -> Result<Vec<BeliefVersion>> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let key = key.to_string();
        let aid = agent_id.map(str::to_string);

        run_on_os_thread(move || -> Result<Vec<BeliefVersion>> {
            client.run(|client| {
            let stmt = format!(
                "SELECT id, key, content, source, valid_from, valid_to, superseded_by
                 FROM {qualified_table}
                 WHERE key = $1
                   AND agent_id = COALESCE($2, (SELECT id FROM {qualified_agents} WHERE alias = 'default' LIMIT 1))
                 ORDER BY valid_from ASC"
            );
            let rows = client.query(&stmt, &[&key, &aid])?;
            Ok(rows
                .iter()
                .map(|r| {
                    let vf: DateTime<Utc> = r.get("valid_from");
                    let vt: Option<DateTime<Utc>> = r.get("valid_to");
                    BeliefVersion {
                        id: r.get("id"),
                        key: r.get("key"),
                        content: r.get("content"),
                        source: r.get("source"),
                        valid_from: vf.to_rfc3339(),
                        valid_to: vt.map(|t| t.to_rfc3339()),
                        superseded_by: r.get("superseded_by"),
                    }
                })
                .collect())
            })
        })
        .await
    }

    /// What did the agent believe at time `asof`? Applicative bitemporality:
    /// returns the versions whose validity window contains `asof`
    /// (`valid_from <= asof < valid_to`), ranked by the same hybrid
    /// keyword+vector fusion as [`Memory::recall`].
    ///
    /// This works for any `asof` since the schema was created — unlike
    /// [`Self::get_asof_system_time`], it does not depend on MVCC garbage
    /// collection.
    pub async fn recall_asof(
        &self,
        query: &str,
        asof_rfc3339: &str,
        limit: usize,
        agent_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        // Validate the timestamp before it goes anywhere near SQL.
        let asof: DateTime<Utc> = asof_rfc3339
            .parse()
            .context("asof must be an RFC 3339 timestamp")?;
        let asof_s = asof.to_rfc3339();

        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let query = normalize_recent_recall_query(query).trim().to_string();
        let query_vec = self.embed_query(&query).await;
        let aid = agent_id.map(str::to_string);

        run_on_os_thread(move || -> Result<Vec<MemoryEntry>> {
            client.run(|client| {
            let columns = CockroachMemory::entry_columns();
            #[allow(clippy::cast_possible_wrap)]
            let limit_i64 = limit as i64;
            // $4: optional agent scope — readers must be scopeable or one
            // MCP server can read another agent's memories.
            let window = "m.valid_from <= $2::TEXT::TIMESTAMPTZ
                          AND (m.valid_to IS NULL OR m.valid_to > $2::TEXT::TIMESTAMPTZ)
                          AND ($4::TEXT IS NULL OR m.agent_id = $4)";

            // Keyword arms run plainto_tsquery('simple', $1), which
            // ERRORS on a lexeme-free query (punctuation-only) instead of
            // matching nothing — so a lexeme-free query must never reach
            // them. Vector recall still works for such queries.
            let has_lexemes = query.chars().any(char::is_alphanumeric);
            if !has_lexemes && query_vec.is_none() {
                let stmt = format!(
                    "SELECT {columns}, 0.0::FLOAT8 AS score
                     FROM {qualified_table} m
                     LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                     WHERE ($1 = '' OR true) AND {window}
                     ORDER BY m.valid_from DESC
                     LIMIT $3"
                );
                let rows = client.query(&stmt, &[&query, &asof_s, &limit_i64, &aid])?;
                return rows
                    .iter()
                    .map(CockroachMemory::row_to_entry)
                    .collect::<Result<Vec<MemoryEntry>>>();
            }

            let keyword_stmt = format!(
                "SELECT {columns},
                        (
                          CASE WHEN to_tsvector('simple', translate(m.key, '_-', '  ')) @@ plainto_tsquery('simple', $1)
                            THEN ts_rank(to_tsvector('simple', translate(m.key, '_-', '  ')), plainto_tsquery('simple', $1)) * 2.0
                            ELSE 0.0 END +
                          CASE WHEN to_tsvector('simple', m.content) @@ plainto_tsquery('simple', $1)
                            THEN ts_rank(to_tsvector('simple', m.content), plainto_tsquery('simple', $1))
                            ELSE 0.0 END
                        )::FLOAT8 AS score
                 FROM {qualified_table} m
                 LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                 WHERE to_tsvector('simple', translate(m.key, '_-', '  ') || ' ' || m.content) @@ plainto_tsquery('simple', $1)
                   AND {window}
                 ORDER BY score DESC, m.valid_from DESC
                 LIMIT $3"
            );
            let keyword_rows: Vec<Row> = if !has_lexemes {
                Vec::new()
            } else {
                client.query(&keyword_stmt, &[&query, &asof_s, &limit_i64, &aid])?
            };

            let vector_rows: Vec<Row> = if let Some(qv) = query_vec.as_ref() {
                let qlit = CockroachMemory::vector_literal(qv);
                let stmt = format!(
                    "SELECT {columns},
                            (1.0 - (m.embedding <=> $1::TEXT::vector))::FLOAT8 AS score
                     FROM {qualified_table} m
                     LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                     WHERE m.embedding IS NOT NULL AND {window}
                     ORDER BY m.embedding <=> $1::TEXT::vector
                     LIMIT $3"
                );
                client.query(&stmt, &[&qlit, &asof_s, &limit_i64, &aid])?
            } else {
                Vec::new()
            };

            let mut entries: HashMap<String, MemoryEntry> = HashMap::new();
            let mut keyword_scores: Vec<(String, f32)> = Vec::new();
            for row in &keyword_rows {
                let entry = CockroachMemory::row_to_entry(row)?;
                #[allow(clippy::cast_possible_truncation)]
                let s = entry.score.unwrap_or(0.0) as f32;
                if s > 0.0 {
                    keyword_scores.push((entry.id.clone(), s));
                }
                entries.insert(entry.id.clone(), entry);
            }
            let mut vector_scores: Vec<(String, f32)> = Vec::new();
            for row in &vector_rows {
                let entry = CockroachMemory::row_to_entry(row)?;
                #[allow(clippy::cast_possible_truncation)]
                let s = entry.score.unwrap_or(0.0) as f32;
                vector_scores.push((entry.id.clone(), s));
                entries.entry(entry.id.clone()).or_insert(entry);
            }

            let merged = hybrid_merge(
                &vector_scores,
                &keyword_scores,
                VECTOR_WEIGHT,
                KEYWORD_WEIGHT,
                limit,
            );
            Ok(merged
                .into_iter()
                .filter_map(|scored| {
                    entries.remove(&scored.id).map(|mut e| {
                        e.score = Some(f64::from(scored.final_score));
                        e
                    })
                })
                .collect())
            })
        })
        .await
    }

    /// MVCC time-travel read of a key's current row *as the database itself
    /// stored it* at `asof`, via CockroachDB's `AS OF SYSTEM TIME`.
    ///
    /// Bounded by the cluster's MVCC garbage-collection window
    /// (`gc.ttlseconds`, measured **14400 s = 4 h** on both the local v26.2.5
    /// cluster and typical defaults) and errors if `asof` predates the
    /// schema's creation. The applicative window ([`Self::recall_asof`])
    /// covers the unbounded horizon; this covers the recent one from the
    /// storage engine's own history — the two must agree, which is exactly
    /// what makes it a good audit cross-check.
    pub async fn get_asof_system_time(
        &self,
        key: &str,
        asof_rfc3339: &str,
    ) -> Result<Option<MemoryEntry>> {
        let asof: DateTime<Utc> = asof_rfc3339
            .parse()
            .context("asof must be an RFC 3339 timestamp")?;
        // Parsed and re-serialized before inlining: injection-safe.
        let asof_lit = asof.to_rfc3339_opts(SecondsFormat::Micros, true);

        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let key = key.to_string();

        run_on_os_thread(move || -> Result<Option<MemoryEntry>> {
            client.run(|client| {
            // No agents JOIN: AS OF SYSTEM TIME applies per-statement and the
            // single-table form is the one CockroachDB documents.
            let stmt = format!(
                "SELECT m.id, m.key, m.content, m.category, m.created_at,
                        m.session_id, m.namespace, m.importance, m.superseded_by,
                        m.kind, m.pinned, m.tenant_id,
                        NULL::TEXT AS agent_alias, m.agent_id
                 FROM {qualified_table} m AS OF SYSTEM TIME '{asof_lit}'
                 WHERE m.key = $1 AND m.valid_to IS NULL
                 LIMIT 1"
            );
            let row = client.query_opt(&stmt, &[&key])?;
            row.as_ref().map(CockroachMemory::row_to_entry).transpose()
            })
        })
        .await
    }

    /// All ledger entries, grouped by chain, oldest first within each chain.
    pub async fn ledger_entries(&self) -> Result<Vec<LedgerEntry>> {
        let client = self.client.get().clone();
        let qualified_ledger = self.qualified_ledger.clone();

        run_on_os_thread(move || -> Result<Vec<LedgerEntry>> {
            client.run(|client| {
                let stmt = format!(
                    "SELECT chain_id, id, ts, kind, actor, target, params, prev_hmac, hmac
                     FROM {qualified_ledger} ORDER BY chain_id ASC, id ASC"
                );
                let rows = client.query(&stmt, &[])?;
                Ok(rows.iter().map(Self::row_to_ledger_entry).collect())
            })
        })
        .await
    }

    fn row_to_ledger_entry(r: &Row) -> LedgerEntry {
        let ts: DateTime<Utc> = r.get("ts");
        LedgerEntry {
            chain_id: r.get("chain_id"),
            id: r.get("id"),
            ts: ts_micros_string(&ts),
            kind: r.get("kind"),
            actor: r.get("actor"),
            target: r.get("target"),
            params: r.get("params"),
            prev_hmac: r.get("prev_hmac"),
            hmac: r.get("hmac"),
        }
    }

    /// Verify the whole memory against its ledger:
    ///
    /// 1. **Chains**: recompute every entry's HMAC and linkage, per scope
    ///    chain — catches any edit to the ledger itself, including moving an
    ///    entry between chains (`chain_id` is signed).
    /// 2. **Row cross-check**: every memory row's full state hash
    ///    ([`ledger::row_sha256`]: key, content, validity window, importance,
    ///    pinned) must equal the hash in the *latest* ledger entry committing
    ///    to that row — catches direct `UPDATE`s that bypassed the write
    ///    path, including a `valid_to` flip that "resurrects" a superseded
    ///    belief without touching `content`.
    ///
    /// Ledger and rows are read **in one transaction** — one snapshot. Two
    /// separate reads would race concurrent legitimate writes and report
    /// phantom tampering.
    pub async fn verify_ledger(&self) -> Result<LedgerVerification> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();

        run_on_os_thread(move || -> Result<LedgerVerification> {
            client.run(|client| {
                type RowState = (String, String, String, Option<f64>, bool, DateTime<Utc>, Option<DateTime<Utc>>);
                let (entries, rows): (Vec<LedgerEntry>, Vec<RowState>) = with_txn_retry(|| {
                    let mut tx = client.transaction()?;
                    let e_stmt = format!(
                        "SELECT chain_id, id, ts, kind, actor, target, params, prev_hmac, hmac
                         FROM {qualified_ledger} ORDER BY chain_id ASC, id ASC"
                    );
                    let entries: Vec<LedgerEntry> = tx
                        .query(&e_stmt, &[])?
                        .iter()
                        .map(Self::row_to_ledger_entry)
                        .collect();
                    let r_stmt = format!(
                        "SELECT id, key, content, importance, pinned, valid_from, valid_to
                         FROM {qualified_table}"
                    );
                    let rows: Vec<RowState> = tx
                        .query(&r_stmt, &[])?
                        .iter()
                        .map(|r| {
                            (
                                r.get(0),
                                r.get(1),
                                r.get(2),
                                r.get(3),
                                r.get(4),
                                r.get(5),
                                r.get(6),
                            )
                        })
                        .collect();
                    tx.commit()?;
                    Ok((entries, rows))
                })?;

                // Per-chain HMAC verification.
                let mut chains: Vec<ChainVerification> = Vec::new();
                let mut start = 0usize;
                while start < entries.len() {
                    let chain_id = entries[start].chain_id.clone();
                    let end = entries[start..]
                        .iter()
                        .position(|e| e.chain_id != chain_id)
                        .map_or(entries.len(), |p| start + p);
                    let report = ledger::verify_chain(&hmac_key, &entries[start..end]);
                    chains.push(ChainVerification { chain_id, report });
                    start = end;
                }

                // Latest committed row hash per row id. RFC 3339 UTC strings
                // sort lexicographically = chronologically, so `ts` orders
                // commitments across chains.
                let mut expected: HashMap<String, (String, String)> = HashMap::new();
                let mut commit = |id: &str, ts: &str, sha: &str| {
                    match expected.get(id) {
                        Some((prev_ts, _)) if prev_ts.as_str() >= ts => {}
                        _ => {
                            expected.insert(id.to_string(), (ts.to_string(), sha.to_string()));
                        }
                    }
                };
                for e in &entries {
                    if let Some(sha) = e.params.get("row_sha256").and_then(|v| v.as_str()) {
                        commit(&e.target, &e.ts, sha);
                    }
                    if let (Some(old_id), Some(old_sha)) = (
                        e.params.get("supersedes").and_then(|v| v.as_str()),
                        e.params
                            .get("superseded_row_sha256")
                            .and_then(|v| v.as_str()),
                    ) {
                        commit(old_id, &e.ts, old_sha);
                    }
                }

                let rows_checked = rows.len();
                let mut row_mismatches = Vec::new();
                for (id, key, content, importance, pinned, vf, vt) in rows {
                    let vt_s = vt.map(|t| ts_micros_string(&t));
                    let row_sha = row_sha256(
                        &key,
                        &content,
                        &ts_micros_string(&vf),
                        vt_s.as_deref(),
                        importance,
                        pinned,
                    );
                    match expected.get(&id) {
                        None => {
                            // Row with no ledger commitment at all: inserted
                            // around the write path — flag it.
                            row_mismatches.push(RowMismatch {
                                id,
                                key,
                                ledger_sha256: "<no ledger entry>".into(),
                                row_sha256: row_sha,
                            });
                        }
                        Some((_, ledger_sha)) if ledger_sha != &row_sha => {
                            row_mismatches.push(RowMismatch {
                                id,
                                key,
                                ledger_sha256: ledger_sha.clone(),
                                row_sha256: row_sha,
                            });
                        }
                        Some(_) => {}
                    }
                }

                Ok(LedgerVerification {
                    chains,
                    row_mismatches,
                    rows_checked,
                })
            })
        })
        .await
    }
}

#[async_trait]
impl Memory for CockroachMemory {
    fn name(&self) -> &str {
        "cockroach"
    }

    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> Result<()> {
        self.store_with_agent(key, content, category, session_id, None, None, None)
            .await
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        self.recall_hybrid(None, query, limit, session_id, since, until)
            .await
    }

    async fn get(&self, key: &str) -> Result<Option<MemoryEntry>> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let key = key.to_string();

        run_on_os_thread(move || -> Result<Option<MemoryEntry>> {
            client.run(|client| {
            let columns = CockroachMemory::entry_columns();
            let stmt = format!(
                "
                SELECT {columns}
                FROM {qualified_table} m
                LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                WHERE m.key = $1 AND m.valid_to IS NULL
                LIMIT 1
                "
            );

            let row = client.query_opt(&stmt, &[&key])?;
            row.as_ref().map(CockroachMemory::row_to_entry).transpose()
            })
        })
        .await
    }

    async fn get_for_agent(&self, key: &str, agent_id: &str) -> Result<Option<MemoryEntry>> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let key = key.to_string();
        let agent_id = agent_id.to_string();

        run_on_os_thread(move || -> Result<Option<MemoryEntry>> {
            client.run(|client| {
            let columns = CockroachMemory::entry_columns();
            let stmt = format!(
                "
                SELECT {columns}
                FROM {qualified_table} m
                LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                WHERE m.key = $1 AND m.agent_id = $2 AND m.valid_to IS NULL
                LIMIT 1
                "
            );

            let row = client.query_opt(&stmt, &[&key, &agent_id])?;
            row.as_ref().map(CockroachMemory::row_to_entry).transpose()
            })
        })
        .await
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let category = category.map(Self::category_to_str);
        let sid = session_id.map(sanitize_session_key);

        run_on_os_thread(move || -> Result<Vec<MemoryEntry>> {
            client.run(|client| {
            let columns = CockroachMemory::entry_columns();
            let stmt = format!(
                "
                SELECT {columns}
                FROM {qualified_table} m
                LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                WHERE ($1::TEXT IS NULL OR m.category = $1)
                  AND ($2::TEXT IS NULL OR m.session_id = $2)
                  AND m.valid_to IS NULL
                ORDER BY m.updated_at DESC
                "
            );

            let category_ref = category.as_deref();
            let session_ref = sid.as_deref();
            let rows = client.query(&stmt, &[&category_ref, &session_ref])?;
            rows.iter()
                .map(CockroachMemory::row_to_entry)
                .collect::<Result<Vec<MemoryEntry>>>()
            })
        })
        .await
    }

    async fn forget(&self, key: &str) -> Result<bool> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let key = key.to_string();

        run_on_os_thread(move || -> Result<bool> {
            client.run(|client| {
            let stmt = format!("DELETE FROM {qualified_table} WHERE key = $1 RETURNING id");
            let deleted = with_txn_retry(|| {
                let mut tx = client.transaction()?;
                let ids: Vec<String> = tx
                    .query(&stmt, &[&key])?
                    .iter()
                    .map(|r| r.get::<_, String>(0))
                    .collect();
                if !ids.is_empty() {
                    ledger_append_in_tx(
                        &mut tx,
                        &qualified_ledger,
                        &hmac_key,
                        SYSTEM_CHAIN,
                        "forget",
                        "system",
                        "*",
                        json!({ "key": key, "deleted_ids": ids }),
                    )?;
                }
                let n = ids.len();
                tx.commit()?;
                Ok(n)
            })?;
            Ok(deleted > 0)
            })
        })
        .await
    }

    async fn forget_for_agent(&self, key: &str, agent_id: &str) -> Result<bool> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let key = key.to_string();
        let agent_id = agent_id.to_string();

        run_on_os_thread(move || -> Result<bool> {
            client.run(|client| {
            let stmt = format!(
                "DELETE FROM {qualified_table} WHERE key = $1 AND agent_id = $2 RETURNING id"
            );
            let deleted = with_txn_retry(|| {
                let mut tx = client.transaction()?;
                let ids: Vec<String> = tx
                    .query(&stmt, &[&key, &agent_id])?
                    .iter()
                    .map(|r| r.get::<_, String>(0))
                    .collect();
                if !ids.is_empty() {
                    ledger_append_in_tx(
                        &mut tx,
                        &qualified_ledger,
                        &hmac_key,
                        SYSTEM_CHAIN,
                        "forget",
                        &agent_id,
                        "*",
                        json!({ "key": key, "deleted_ids": ids }),
                    )?;
                }
                let n = ids.len();
                tx.commit()?;
                Ok(n)
            })?;
            Ok(deleted > 0)
            })
        })
        .await
    }

    async fn purge_namespace(&self, namespace: &str) -> Result<usize> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let namespace = namespace.to_string();

        run_on_os_thread(move || -> Result<usize> {
            client.run(|client| {
            // Mass deletions must be ledgered in the same transaction —
            // an unledgered purge is indistinguishable from tampering.
            let stmt =
                format!("DELETE FROM {qualified_table} WHERE namespace = $1 RETURNING id");
            let deleted = with_txn_retry(|| {
                let mut tx = client.transaction()?;
                let ids: Vec<String> =
                    tx.query(&stmt, &[&namespace])?.iter().map(|r| r.get(0)).collect();
                if !ids.is_empty() {
                    ledger_append_in_tx(
                        &mut tx,
                        &qualified_ledger,
                        &hmac_key,
                        SYSTEM_CHAIN,
                        "purge_namespace",
                        "system",
                        "*",
                        json!({ "namespace": namespace, "count": ids.len(), "deleted_ids": ids }),
                    )?;
                }
                let n = ids.len();
                tx.commit()?;
                Ok(n)
            })?;
            Ok(deleted)
            })
        })
        .await
    }

    async fn purge_session(&self, session_id: &str) -> Result<usize> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        // Sanitized on write, so it must be sanitized on delete too or the
        // filter silently misses every row (round-trip asymmetry).
        let session_id = sanitize_session_key(session_id);

        run_on_os_thread(move || -> Result<usize> {
            client.run(|client| {
            let stmt =
                format!("DELETE FROM {qualified_table} WHERE session_id = $1 RETURNING id");
            let deleted = with_txn_retry(|| {
                let mut tx = client.transaction()?;
                let ids: Vec<String> =
                    tx.query(&stmt, &[&session_id])?.iter().map(|r| r.get(0)).collect();
                if !ids.is_empty() {
                    ledger_append_in_tx(
                        &mut tx,
                        &qualified_ledger,
                        &hmac_key,
                        SYSTEM_CHAIN,
                        "purge_session",
                        "system",
                        "*",
                        json!({ "session_id": session_id, "count": ids.len(), "deleted_ids": ids }),
                    )?;
                }
                let n = ids.len();
                tx.commit()?;
                Ok(n)
            })?;
            Ok(deleted)
            })
        })
        .await
    }

    async fn purge_session_for_agent(&self, session_id: &str, agent_id: &str) -> Result<usize> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let session_id = sanitize_session_key(session_id);
        let agent_id = agent_id.to_string();

        run_on_os_thread(move || -> Result<usize> {
            client.run(|client| {
            let stmt = format!(
                "DELETE FROM {qualified_table} WHERE session_id = $1 AND agent_id = $2 RETURNING id"
            );
            let deleted = with_txn_retry(|| {
                let mut tx = client.transaction()?;
                let ids: Vec<String> = tx
                    .query(&stmt, &[&session_id, &agent_id])?
                    .iter()
                    .map(|r| r.get(0))
                    .collect();
                if !ids.is_empty() {
                    ledger_append_in_tx(
                        &mut tx,
                        &qualified_ledger,
                        &hmac_key,
                        SYSTEM_CHAIN,
                        "purge_session",
                        &agent_id,
                        "*",
                        json!({ "session_id": session_id, "count": ids.len(), "deleted_ids": ids }),
                    )?;
                }
                let n = ids.len();
                tx.commit()?;
                Ok(n)
            })?;
            Ok(deleted)
            })
        })
        .await
    }

    async fn purge_agent(&self, agent_alias: &str) -> Result<usize> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let alias = agent_alias.to_string();

        run_on_os_thread(move || -> Result<usize> {
            client.run(|client| {
            let stmt = format!(
                "DELETE FROM {qualified_table} WHERE agent_id = (SELECT id FROM {qualified_agents} WHERE alias = $1) RETURNING id"
            );
            let deleted = with_txn_retry(|| {
                let mut tx = client.transaction()?;
                let ids: Vec<String> =
                    tx.query(&stmt, &[&alias])?.iter().map(|r| r.get(0)).collect();
                if !ids.is_empty() {
                    ledger_append_in_tx(
                        &mut tx,
                        &qualified_ledger,
                        &hmac_key,
                        SYSTEM_CHAIN,
                        "purge_agent",
                        &alias,
                        "*",
                        json!({ "agent_alias": alias, "count": ids.len(), "deleted_ids": ids }),
                    )?;
                }
                let n = ids.len();
                tx.commit()?;
                Ok(n)
            })?;
            Ok(deleted)
            })
        })
        .await
    }

    async fn export_agent(&self, agent_alias: &str) -> Result<Vec<MemoryEntry>> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let alias = agent_alias.to_string();

        run_on_os_thread(move || -> Result<Vec<MemoryEntry>> {
            client.run(|client| {
            let columns = CockroachMemory::entry_columns();
            let stmt = format!(
                "
                SELECT {columns}
                FROM {qualified_table} m
                LEFT JOIN {qualified_agents} a ON a.id = m.agent_id
                WHERE m.agent_id = (SELECT id FROM {qualified_agents} WHERE alias = $1)
                ORDER BY m.created_at ASC
                "
            );
            let rows = client.query(&stmt, &[&alias])?;
            rows.iter()
                .map(CockroachMemory::row_to_entry)
                .collect::<Result<Vec<MemoryEntry>>>()
            })
        })
        .await
    }

    async fn rename_agent(&self, from: &str, to: &str) -> Result<usize> {
        let client = self.client.get().clone();
        let qualified_agents = self.qualified_agents.clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let from = from.to_string();
        let to = to.to_string();

        run_on_os_thread(move || -> Result<usize> {
            client.run(|client| {
            let mut tx = client.transaction()?;
            let to_rows: i64 = tx
                .query_one(
                    &format!(
                        "SELECT COUNT(*) FROM {qualified_table} WHERE agent_id = (SELECT id FROM {qualified_agents} WHERE alias = $1)"
                    ),
                    &[&to],
                )?
                .get(0);
            if to_rows > 0 {
                anyhow::bail!(
                    "cannot rename agent memory to `{to}`: an existing memory store under that alias has {to_rows} row(s); refusing to merge"
                );
            }
            tx.execute(
                &format!("DELETE FROM {qualified_agents} WHERE alias = $1"),
                &[&to],
            )?;
            let updated = tx.execute(
                &format!("UPDATE {qualified_agents} SET alias = $2 WHERE alias = $1"),
                &[&from, &to],
            )?;
            if updated > 0 {
                // Attribution changes are mutations too: an unledgered rename
                // would let history be silently re-attributed.
                ledger_append_in_tx(
                    &mut tx,
                    &qualified_ledger,
                    &hmac_key,
                    SYSTEM_CHAIN,
                    "rename_agent",
                    "system",
                    "*",
                    json!({ "from": from, "to": to }),
                )?;
            }
            tx.commit()?;
            usize::try_from(updated).context("CockroachDB returned an oversized update count")
            })
        })
        .await
    }

    async fn count_agent(&self, agent_alias: &str) -> Result<usize> {
        let client = self.client.get().clone();
        let qualified_agents = self.qualified_agents.clone();
        let alias = agent_alias.to_string();

        run_on_os_thread(move || -> Result<usize> {
            client.run(|client| {
            // Mirrors `rename_agent`: residue is the alias row (0 or 1).
            let stmt = format!("SELECT COUNT(*) FROM {qualified_agents} WHERE alias = $1");
            let row = client.query_one(&stmt, &[&alias])?;
            let count: i64 = row.get(0);
            usize::try_from(count).context("CockroachDB returned an oversized agent count")
            })
        })
        .await
    }

    async fn count(&self) -> Result<usize> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();

        run_on_os_thread(move || -> Result<usize> {
            client.run(|client| {
            let stmt = format!("SELECT COUNT(*) FROM {qualified_table}");
            let count: i64 = client.query_one(&stmt, &[])?.get(0);
            let count =
                usize::try_from(count).context("CockroachDB returned a negative memory count")?;
            Ok(count)
            })
        })
        .await
    }

    async fn health_check(&self) -> bool {
        let client = self.client.get().clone();
        run_on_os_thread(move || {
            Ok(client
                .run(|c| c.simple_query("SELECT 1").map_err(anyhow::Error::from))
                .is_ok())
        })
        .await
        .unwrap_or(false)
    }

    async fn supersede(&self, superseded_ids: &[String], new_id: &str) -> Result<()> {
        if superseded_ids.is_empty() {
            return Ok(());
        }
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let ids: Vec<String> = superseded_ids.to_vec();
        let new_id = new_id.to_string();

        run_on_os_thread(move || -> Result<()> {
            client.run(|client| {
            let stmt = format!(
                "UPDATE {qualified_table} SET superseded_by = $1, updated_at = NOW() WHERE id = ANY($2)"
            );
            with_txn_retry(|| {
                let mut tx = client.transaction()?;
                let updated = tx.execute(&stmt, &[&new_id, &ids])?;
                if updated > 0 {
                    ledger_append_in_tx(
                        &mut tx,
                        &qualified_ledger,
                        &hmac_key,
                        SYSTEM_CHAIN,
                        "supersede_mark",
                        "system",
                        &new_id,
                        json!({ "superseded_ids": ids, "count": updated }),
                    )?;
                }
                tx.commit()
            })?;
            Ok(())
            })
        })
        .await
    }

    async fn count_in_scope(
        &self,
        namespace: Option<&str>,
        category: Option<&MemoryCategory>,
    ) -> Result<u64> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let namespace = namespace.map(str::to_string);
        let category = category.map(Self::category_to_str);

        run_on_os_thread(move || -> Result<u64> {
            client.run(|client| {
            let stmt = format!(
                "SELECT COUNT(*) FROM {qualified_table}
                 WHERE ($1::TEXT IS NULL OR namespace = $1)
                   AND ($2::TEXT IS NULL OR category = $2)"
            );
            let count: i64 = client
                .query_one(&stmt, &[&namespace.as_deref(), &category.as_deref()])?
                .get(0);
            u64::try_from(count).context("CockroachDB returned a negative count")
            })
        })
        .await
    }

    async fn stats(&self) -> Result<MemoryStats> {
        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();

        run_on_os_thread(move || -> Result<MemoryStats> {
            client.run(|client| {
            let totals = client.query_one(
                &format!(
                    "SELECT COUNT(*),
                            COUNT(*) FILTER (WHERE superseded_by IS NOT NULL),
                            COUNT(*) FILTER (WHERE pinned)
                     FROM {qualified_table}"
                ),
                &[],
            )?;
            let total_rows: i64 = totals.get(0);
            let superseded_rows: i64 = totals.get(1);
            let pinned_rows: i64 = totals.get(2);

            let by_category = client
                .query(
                    &format!(
                        "SELECT category, COUNT(*) FROM {qualified_table} GROUP BY category ORDER BY category"
                    ),
                    &[],
                )?
                .iter()
                .map(|r| {
                    let cat: String = r.get(0);
                    let n: i64 = r.get(1);
                    (cat, n.max(0) as u64)
                })
                .collect();

            Ok(MemoryStats {
                total_rows: total_rows.max(0) as u64,
                by_category,
                superseded_rows: superseded_rows.max(0) as u64,
                pinned_rows: pinned_rows.max(0) as u64,
                bytes: 0,
            })
            })
        })
        .await
    }

    async fn store_with_options(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        options: StoreOptions,
    ) -> Result<()> {
        self.store_with_options_and_agent(key, content, category, session_id, options, None)
            .await
    }

    async fn store_with_options_and_agent(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        options: StoreOptions,
        agent_id: Option<&str>,
    ) -> Result<()> {
        let embedding = if self.dimensions > 0 {
            // Embed key + content together: keys carry real semantics
            // ("deploy_runbook") and must be reachable by vector recall.
            let embed_text = format!("{key}\n{content}");
            match self.embedder.embed_one(&embed_text).await {
                Ok(v) if v.len() == self.dimensions => Some(Self::vector_literal(&v)),
                Ok(_) => {
                    tracing::warn!("content embedding has wrong dimensionality; storing without");
                    None
                }
                Err(err) => {
                    tracing::warn!("content embedding failed ({err}); storing without");
                    None
                }
            }
        } else {
            None
        };

        let client = self.client.get().clone();
        let qualified_table = self.qualified_table.clone();
        let qualified_agents = self.qualified_agents.clone();
        let qualified_ledger = self.qualified_ledger.clone();
        let hmac_key = self.hmac_key.clone();
        let key = key.to_string();
        let content = content.to_string();
        let category = Self::category_to_str(&category);
        let sid = session_id.map(sanitize_session_key);
        let aid = agent_id.map(str::to_string);
        let namespace = options.namespace.unwrap_or_else(|| "default".into());
        let importance = options.importance;
        let kind = options
            .kind
            .as_ref()
            .map(|k| serde_json::to_string(k))
            .transpose()?;
        let pinned = options.pinned;
        let tenant_id = options.tenant_id.unwrap_or_else(|| "default".into());

        let has_vector = self.dimensions > 0;
        run_on_os_thread(move || -> Result<()> {
            client.run(|client| {
            // The embedding column only exists when the backend was created
            // with a real embedder. `$14::TEXT::vector` (not `$14::vector`)
            // keeps the wire parameter type TEXT — rust-postgres cannot
            // serialize into CockroachDB's `vector` type directly.
            let (embedding_col, embedding_val, embedding_upd) = if has_vector {
                (", embedding", ", $14::TEXT::vector", ",\n                    embedding = EXCLUDED.embedding")
            } else {
                ("", "", "")
            };
            // In-place upsert of the *current* version — the arbiter is the
            // partial unique index, so belief history rows never conflict.
            // This is the legacy non-versioned surface; `store_belief` /
            // `supersede_belief` is the versioned one. valid_from is kept on
            // update (an overwrite is not a supersession).
            let stmt = format!(
                "
                INSERT INTO {qualified_table}
                    (id, tenant_id, agent_id, namespace, key, content, category,
                     kind, importance, pinned, session_id, created_at, updated_at,
                     valid_from{embedding_col})
                VALUES
                    ($1, $2,
                     COALESCE($3, (SELECT id FROM {qualified_agents} WHERE alias = 'default' LIMIT 1)),
                     $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $12{embedding_val})
                ON CONFLICT (agent_id, key) WHERE valid_to IS NULL DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    kind = EXCLUDED.kind,
                    importance = EXCLUDED.importance,
                    pinned = EXCLUDED.pinned,
                    namespace = EXCLUDED.namespace,
                    tenant_id = EXCLUDED.tenant_id,
                    session_id = EXCLUDED.session_id,
                    updated_at = EXCLUDED.updated_at{embedding_upd}
                RETURNING id, agent_id, valid_from, valid_to
                "
            );

            let id = Uuid::new_v4().to_string();
            let content_hash = content_sha256(&content);
            with_txn_retry(|| {
                let now = now_micros();
                let mut params: Vec<&(dyn postgres::types::ToSql + Sync)> = vec![
                    &id,
                    &tenant_id,
                    &aid,
                    &namespace,
                    &key,
                    &content,
                    &category,
                    &kind,
                    &importance,
                    &pinned,
                    &sid,
                    &now,
                    &now,
                ];
                if has_vector {
                    params.push(&embedding);
                }
                let mut tx = client.transaction()?;
                // RETURNING gives the row as stored: on conflict-update the
                // original valid_from survives, and the ledger hash must
                // commit to that, not to this call's timestamp.
                let row = tx.query_one(&stmt, &params)?;
                let row_id: String = row.get(0);
                let row_agent: String = row.get(1);
                let vf: DateTime<Utc> = row.get(2);
                let vt: Option<DateTime<Utc>> = row.get(3);
                let vt_s = vt.map(|t| ts_micros_string(&t));
                let row_sha = row_sha256(
                    &key,
                    &content,
                    &ts_micros_string(&vf),
                    vt_s.as_deref(),
                    importance,
                    pinned,
                );
                ledger_append_in_tx(
                    &mut tx,
                    &qualified_ledger,
                    &hmac_key,
                    &chain_for(&tenant_id, &row_agent),
                    "store",
                    &row_agent,
                    &row_id,
                    json!({ "key": key, "content_sha256": content_hash, "row_sha256": row_sha }),
                )?;
                tx.commit()
            })?;
            Ok(())
            })
        })
        .await
    }

    async fn store_with_agent(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        namespace: Option<&str>,
        importance: Option<f64>,
        agent_id: Option<&str>,
    ) -> Result<()> {
        let mut options = StoreOptions::default();
        options.namespace = namespace.map(str::to_string);
        options.importance = importance;
        self.store_with_options_and_agent(key, content, category, session_id, options, agent_id)
            .await
    }

    async fn recall_for_agents(
        &self,
        allowed_agent_ids: &[&str],
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> Result<Vec<MemoryEntry>> {
        if allowed_agent_ids.is_empty() {
            return self.recall(query, limit, session_id, since, until).await;
        }
        let allowed: Vec<String> = allowed_agent_ids.iter().map(|s| (*s).to_string()).collect();
        self.recall_hybrid(Some(allowed), query, limit, session_id, since, until)
            .await
    }

    async fn ensure_agent_uuid(&self, alias: &str) -> Result<String> {
        let client = self.client.get().clone();
        let qualified_agents = self.qualified_agents.clone();
        let alias = alias.to_string();
        run_on_os_thread(move || -> Result<String> {
            client.run(|client| {
            let candidate = Uuid::new_v4().to_string();
            let insert = format!(
                "INSERT INTO {qualified_agents} (id, alias, created_at)
                 VALUES ($1, $2, NOW())
                 ON CONFLICT (alias) DO NOTHING"
            );
            with_txn_retry(|| client.execute(&insert, &[&candidate, &alias]))?;
            let row: String = client
                .query_one(
                    &format!("SELECT id FROM {qualified_agents} WHERE alias = $1 LIMIT 1"),
                    &[&alias],
                )?
                .get(0);
            Ok(row)
            })
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_identifiers_pass_validation() {
        assert!(validate_identifier("public", "schema").is_ok());
        assert!(validate_identifier("_memories_01", "table").is_ok());
    }

    #[test]
    fn invalid_identifiers_are_rejected() {
        assert!(validate_identifier("", "schema").is_err());
        assert!(validate_identifier("1bad", "schema").is_err());
        assert!(validate_identifier("bad-name", "table").is_err());
    }

    #[test]
    fn parse_category_maps_known_and_custom_values() {
        assert_eq!(
            CockroachMemory::parse_category("core"),
            MemoryCategory::Core
        );
        assert_eq!(
            CockroachMemory::parse_category("custom_notes"),
            MemoryCategory::Custom("custom_notes".into())
        );
    }

    #[test]
    fn vector_literal_formats_bracketed_csv() {
        assert_eq!(
            CockroachMemory::vector_literal(&[1.0, -0.5, 0.25]),
            "[1,-0.5,0.25]"
        );
        assert_eq!(CockroachMemory::vector_literal(&[]), "[]");
    }
}
