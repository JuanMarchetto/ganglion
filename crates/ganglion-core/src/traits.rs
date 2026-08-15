//! Core memory trait and entry types.
//!
//! Vendored from zeroclaw (`zeroclaw-api/src/memory_traits.rs` and
//! `zeroclaw-api/src/session_keys.rs`, MIT OR Apache-2.0) and trimmed for
//! Ganglion: the `Attributable` supertrait, `MemoryStrategy`, and the
//! provider-coupled surface were removed. See ATTRIBUTION.md.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Filter criteria for bulk memory export (GDPR Art. 20 data portability).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExportFilter {
    pub namespace: Option<String>,
    pub session_id: Option<String>,
    pub category: Option<MemoryCategory>,
    /// RFC 3339 lower bound (inclusive) on created_at.
    pub since: Option<String>,
    /// RFC 3339 upper bound (inclusive) on created_at.
    pub until: Option<String>,
}

/// A single memory entry
#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub key: String,
    pub content: String,
    pub category: MemoryCategory,
    pub timestamp: String,
    pub session_id: Option<String>,
    pub score: Option<f64>,
    /// Namespace for isolation between agents/contexts.
    #[serde(default = "default_namespace")]
    pub namespace: String,
    /// Importance score (0.0–1.0) for prioritized retrieval.
    #[serde(default)]
    pub importance: Option<f64>,
    /// If this entry was superseded by a newer conflicting entry.
    #[serde(default)]
    pub superseded_by: Option<String>,
    /// Memory kind, orthogonal to the durability/recency category.
    #[serde(default)]
    pub kind: Option<MemoryKind>,
    /// Whether this entry is protected from budget eviction.
    #[serde(default)]
    pub pinned: bool,
    /// Tenant or end-user scope for multi-user memory isolation.
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// Resolved, human-readable agent alias for this row. SQL-backed stores
    /// produce this via `LEFT JOIN agents ON agents.id = memories.agent_id`.
    #[serde(default)]
    pub agent_alias: Option<String>,
    /// Raw value of the storage layer's agent column (`memories.agent_id`,
    /// a UUID FK to `agents.id`).
    #[serde(default, alias = "agent_id")]
    pub agent_id: Option<String>,
}

fn default_namespace() -> String {
    "default".into()
}

impl std::fmt::Debug for MemoryEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryEntry")
            .field("id", &self.id)
            .field("key", &self.key)
            .field("content", &self.content)
            .field("category", &self.category)
            .field("timestamp", &self.timestamp)
            .field("score", &self.score)
            .field("namespace", &self.namespace)
            .field("importance", &self.importance)
            .field("kind", &self.kind)
            .field("pinned", &self.pinned)
            .field("tenant_id", &self.tenant_id)
            .field("agent_alias", &self.agent_alias)
            .finish_non_exhaustive()
    }
}

/// Memory kind, orthogonal to [`MemoryCategory`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Session or event memory.
    Episodic,
    /// Evergreen semantic memory.
    Semantic(SemanticSubtype),
    /// How-to or process memory.
    Procedural,
}

/// Semantic memory subtypes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticSubtype {
    Preference,
    Fact,
    Decision,
    Entity,
}

/// Memory categories for organization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryCategory {
    /// Long-term facts, preferences, decisions
    Core,
    /// Daily session logs
    Daily,
    /// Conversation context
    Conversation,
    /// User-defined custom category
    Custom(String),
}

impl serde::Serialize for MemoryCategory {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for MemoryCategory {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "core" => Self::Core,
            "daily" => Self::Daily,
            "conversation" => Self::Conversation,
            _ => Self::Custom(s),
        })
    }
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "core"),
            Self::Daily => write!(f, "daily"),
            Self::Conversation => write!(f, "conversation"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// Returns true when a recall query should be interpreted as recent/time-only recall.
///
/// A bare "*" is intentionally equivalent to an omitted query for tool-call
/// compatibility. Non-bare wildcard terms such as "wild*" remain keyword queries.
pub fn is_recent_recall_query(query: &str) -> bool {
    let trimmed = query.trim();
    trimmed.is_empty() || trimmed == "*"
}

/// Normalizes recent/time-only recall queries to the backend-neutral empty query.
pub fn normalize_recent_recall_query(query: &str) -> &str {
    if is_recent_recall_query(query) {
        ""
    } else {
        query
    }
}

/// Replace every character outside `[A-Za-z0-9_-]` with `_`. Idempotent.
///
/// Callers building session keys must pre-apply this so runtime keys and the
/// `session_id` column in memory backends agree.
pub fn sanitize_session_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// A single message in a conversation trace for procedural memory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProceduralMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Options for storing memory metadata without growing write-method arity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoreOptions {
    pub namespace: Option<String>,
    pub importance: Option<f64>,
    pub kind: Option<MemoryKind>,
    pub pinned: bool,
    pub tenant_id: Option<String>,
}

impl StoreOptions {
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = Some(importance);
        self
    }

    pub fn with_kind(mut self, kind: MemoryKind) -> Self {
        self.kind = Some(kind);
        self
    }

    pub fn pinned(mut self, pinned: bool) -> Self {
        self.pinned = pinned;
        self
    }

    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn requires_full_options_storage(&self) -> bool {
        self.kind.is_some() || self.pinned || self.tenant_id.is_some()
    }
}

/// Read-side memory store telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_rows: u64,
    pub by_category: Vec<(String, u64)>,
    pub superseded_rows: u64,
    pub pinned_rows: u64,
    pub bytes: u64,
}

/// Core memory trait — implement for any persistence backend
#[async_trait]
pub trait Memory: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Store a memory entry, optionally scoped to a session
    async fn store(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
    ) -> anyhow::Result<()>;

    /// Recall memories matching a query (hybrid search), optionally scoped to a
    /// session and time range. Empty, whitespace-only, and bare "*" queries
    /// return recent/time-only entries. Time bounds use RFC 3339 format;
    /// inclusive (created_at >= since, created_at <= until).
    async fn recall(
        &self,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>>;

    /// Get a specific memory by key. Multiple rows may share a `key` (one per
    /// agent); this returns *some* matching row. Agent-scoped callers use
    /// [`get_for_agent`](Self::get_for_agent).
    async fn get(&self, key: &str) -> anyhow::Result<Option<MemoryEntry>>;

    /// Get the memory row matching `(key, agent_id)`.
    async fn get_for_agent(
        &self,
        key: &str,
        agent_id: &str,
    ) -> anyhow::Result<Option<MemoryEntry>> {
        let hit = self.get(key).await?;
        Ok(hit.filter(|e| e.agent_id.as_deref() == Some(agent_id)))
    }

    /// List all memory entries, optionally filtered by category and/or session
    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        session_id: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>>;

    /// Remove a memory by key (every row matching `key`, regardless of agent).
    async fn forget(&self, key: &str) -> anyhow::Result<bool>;

    /// Remove the row matching `(key, agent_id)`; siblings under other agents
    /// are untouched. Returns `true` if a row was removed.
    async fn forget_for_agent(&self, key: &str, agent_id: &str) -> anyhow::Result<bool>;

    /// Remove all memories whose `namespace` equals the given value.
    async fn purge_namespace(&self, _namespace: &str) -> anyhow::Result<usize> {
        anyhow::bail!("purge_namespace not supported by this memory backend")
    }

    /// Remove all memories in a session.
    async fn purge_session(&self, _session_id: &str) -> anyhow::Result<usize> {
        anyhow::bail!("purge_session not supported by this memory backend")
    }

    /// Remove all memories in a session for one agent.
    async fn purge_session_for_agent(
        &self,
        _session_id: &str,
        _agent_id: &str,
    ) -> anyhow::Result<usize> {
        anyhow::bail!("purge_session_for_agent not supported by this memory backend")
    }

    /// Remove every memory row attributed to the given agent alias.
    async fn purge_agent(&self, _agent_alias: &str) -> anyhow::Result<usize> {
        anyhow::bail!("purge_agent not supported by this memory backend")
    }

    /// Export every memory row attributed to `agent_alias`.
    async fn export_agent(&self, _agent_alias: &str) -> anyhow::Result<Vec<MemoryEntry>> {
        Ok(Vec::new())
    }

    /// Re-point every memory row from the `from` alias to the `to` alias.
    async fn rename_agent(&self, _from: &str, _to: &str) -> anyhow::Result<usize> {
        anyhow::bail!("rename_agent not supported by this memory backend")
    }

    /// Read-only residue probe for the agent-rename cascade. MUST mirror what
    /// `rename_agent` moves: for SQL backends that is the `agents` row (alias
    /// presence, 0 or 1), NOT the memory-row count.
    async fn count_agent(&self, _agent_alias: &str) -> anyhow::Result<usize> {
        Ok(0)
    }

    /// Count total memories
    async fn count(&self) -> anyhow::Result<usize>;

    /// Health check
    async fn health_check(&self) -> bool;

    /// Mark entries as superseded by a newer row.
    async fn supersede(&self, _superseded_ids: &[String], _new_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    /// Store a procedural "how to" trace from a tool-calling turn.
    async fn store_procedural(
        &self,
        _messages: &[ProceduralMessage],
        _session_id: Option<&str>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Count rows within a namespace/category scope.
    async fn count_in_scope(
        &self,
        _namespace: Option<&str>,
        _category: Option<&MemoryCategory>,
    ) -> anyhow::Result<u64> {
        Ok(0)
    }

    /// Read-side memory store telemetry.
    async fn stats(&self) -> anyhow::Result<MemoryStats> {
        Ok(MemoryStats::default())
    }

    /// Recall memories scoped to a specific namespace.
    async fn recall_namespaced(
        &self,
        namespace: &str,
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>> {
        let entries = self
            .recall(query, limit * 2, session_id, since, until)
            .await?;
        let filtered: Vec<MemoryEntry> = entries
            .into_iter()
            .filter(|e| e.namespace == namespace)
            .take(limit)
            .collect();
        Ok(filtered)
    }

    /// Bulk-export memories matching the given filter criteria, ordered by
    /// creation time ascending. Embeddings are excluded.
    async fn export(&self, filter: &ExportFilter) -> anyhow::Result<Vec<MemoryEntry>> {
        let entries = self
            .list(filter.category.as_ref(), filter.session_id.as_deref())
            .await?;
        let filtered: Vec<MemoryEntry> = entries
            .into_iter()
            .filter(|e| {
                if let Some(ref ns) = filter.namespace
                    && e.namespace != *ns
                {
                    return false;
                }
                if let Some(ref since) = filter.since
                    && e.timestamp.as_str() < since.as_str()
                {
                    return false;
                }
                if let Some(ref until) = filter.until
                    && e.timestamp.as_str() > until.as_str()
                {
                    return false;
                }
                true
            })
            .collect();
        Ok(filtered)
    }

    /// Store a memory entry with namespace and importance.
    async fn store_with_metadata(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        _namespace: Option<&str>,
        _importance: Option<f64>,
    ) -> anyhow::Result<()> {
        self.store(key, content, category, session_id).await
    }

    /// Store a memory entry with the full additive metadata surface.
    async fn store_with_options(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        options: StoreOptions,
    ) -> anyhow::Result<()> {
        if options.requires_full_options_storage() {
            anyhow::bail!(
                "memory backend '{}' does not support StoreOptions kind/pinned/tenant_id; use a backend that overrides store_with_options",
                self.name()
            );
        }
        self.store_with_metadata(
            key,
            content,
            category,
            session_id,
            options.namespace.as_deref(),
            options.importance,
        )
        .await
    }

    /// Store a memory entry with full metadata and an explicit agent UUID.
    async fn store_with_options_and_agent(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        options: StoreOptions,
        agent_id: Option<&str>,
    ) -> anyhow::Result<()> {
        if options.requires_full_options_storage() {
            anyhow::bail!(
                "memory backend '{}' does not support agent-attributed StoreOptions kind/pinned/tenant_id; use a backend that overrides store_with_options_and_agent",
                self.name()
            );
        }
        self.store_with_agent(
            key,
            content,
            category,
            session_id,
            options.namespace.as_deref(),
            options.importance,
            agent_id,
        )
        .await
    }

    /// Store a memory entry attributed to an explicit agent UUID. Every
    /// backend must implement this explicitly so the agent_id is never
    /// silently dropped at storage time.
    async fn store_with_agent(
        &self,
        key: &str,
        content: &str,
        category: MemoryCategory,
        session_id: Option<&str>,
        namespace: Option<&str>,
        importance: Option<f64>,
        agent_id: Option<&str>,
    ) -> anyhow::Result<()>;

    /// Recall memory entries scoped to a specific set of agent UUIDs. When
    /// `allowed_agent_ids` is non-empty, the backend filters to rows whose
    /// `agent_id` matches one of the listed UUIDs.
    async fn recall_for_agents(
        &self,
        allowed_agent_ids: &[&str],
        query: &str,
        limit: usize,
        session_id: Option<&str>,
        since: Option<&str>,
        until: Option<&str>,
    ) -> anyhow::Result<Vec<MemoryEntry>>;

    /// Look up (or create) the identifier the backend uses to refer to the
    /// agent named by `alias`. SQL backends return the `agents` row UUID,
    /// inserting if absent.
    async fn ensure_agent_uuid(&self, alias: &str) -> anyhow::Result<String> {
        Ok(alias.to_string())
    }
}
