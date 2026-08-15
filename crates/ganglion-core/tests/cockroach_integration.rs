//! Integration tests for `CockroachMemory` against a real CockroachDB cluster.
//!
//! Behavior extracted from zeroclaw's `sqlite.rs` reference suite (store /
//! recall / get / forget / purge, agent scoping, namespaces, importance,
//! pinned, vector recall) and adapted to the CockroachDB backend.
//!
//! Requires a running cluster (see `deploy/local/up.sh`). Connection string
//! comes from `GANGLION_TEST_DSN`, defaulting to the local haproxy endpoint.
//! Every test uses its own schema, so tests are isolated and parallel-safe.

use ganglion_core::{
    CockroachMemory, HashEmbedding, Memory, MemoryCategory, MemoryKind, SemanticSubtype,
    StoreOptions,
};
use std::sync::Arc;
use std::time::Duration;

fn dsn() -> String {
    std::env::var("GANGLION_TEST_DSN")
        .unwrap_or_else(|_| "postgresql://root@localhost:26257/ganglion?sslmode=disable".into())
}

fn unique_schema() -> String {
    format!("t_{}", uuid::Uuid::new_v4().simple())
}

/// Backend with the deterministic hash embedder (32 dims): hybrid recall on.
fn mem() -> CockroachMemory {
    CockroachMemory::new(
        &dsn(),
        &unique_schema(),
        "memories",
        Some(15),
        Some(Arc::new(HashEmbedding::new(32))),
    )
    .expect("connect + init schema")
}

/// Small pause so `created_at` / `updated_at` ordering is deterministic.
fn tick() {
    std::thread::sleep(Duration::from_millis(30));
}

#[tokio::test(flavor = "multi_thread")]
async fn store_and_get_roundtrip() {
    let m = mem();
    m.store("lang", "Rust is the language", MemoryCategory::Core, None)
        .await
        .unwrap();

    let hit = m.get("lang").await.unwrap().expect("row exists");
    assert_eq!(hit.key, "lang");
    assert_eq!(hit.content, "Rust is the language");
    assert_eq!(hit.category, MemoryCategory::Core);
    assert_eq!(hit.namespace, "default");
    assert_eq!(hit.tenant_id.as_deref(), Some("default"));
    assert!(hit.agent_id.is_some(), "default agent attribution");
    assert_eq!(hit.agent_alias.as_deref(), Some("default"));
}

#[tokio::test(flavor = "multi_thread")]
async fn get_missing_returns_none() {
    let m = mem();
    assert!(m.get("never-stored").await.unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn store_upserts_same_agent_key() {
    let m = mem();
    m.store("k", "first version", MemoryCategory::Core, None)
        .await
        .unwrap();
    m.store("k", "second version", MemoryCategory::Daily, Some("s1"))
        .await
        .unwrap();

    assert_eq!(m.count().await.unwrap(), 1, "upsert must not duplicate");
    let hit = m.get("k").await.unwrap().unwrap();
    assert_eq!(hit.content, "second version");
    assert_eq!(hit.category, MemoryCategory::Daily);
    assert_eq!(hit.session_id.as_deref(), Some("s1"));
}

#[tokio::test(flavor = "multi_thread")]
async fn same_key_isolated_across_agents() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("alpha").await.unwrap();
    let a2 = m.ensure_agent_uuid("beta").await.unwrap();

    m.store_with_agent("k", "alpha's fact", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();
    m.store_with_agent("k", "beta's fact", MemoryCategory::Core, None, None, None, Some(&a2))
        .await
        .unwrap();

    assert_eq!(m.count().await.unwrap(), 2, "one row per agent");
    let h1 = m.get_for_agent("k", &a1).await.unwrap().unwrap();
    let h2 = m.get_for_agent("k", &a2).await.unwrap().unwrap();
    assert_eq!(h1.content, "alpha's fact");
    assert_eq!(h2.content, "beta's fact");
    assert_eq!(h1.agent_alias.as_deref(), Some("alpha"));
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_keyword_finds_content_match() {
    let m = mem();
    m.store("a", "the deployment pipeline uses blue green strategy", MemoryCategory::Core, None)
        .await
        .unwrap();
    m.store("b", "cats prefer boxes over beds", MemoryCategory::Core, None)
        .await
        .unwrap();

    let hits = m.recall("deployment pipeline", 10, None, None, None).await.unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].key, "a");
    assert!(hits[0].score.unwrap_or(0.0) > 0.0, "score populated");
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_key_match_ranks_first() {
    let m = mem();
    m.store("deploy_runbook", "steps to ship", MemoryCategory::Core, None)
        .await
        .unwrap();
    m.store("notes", "we talked about deploy once", MemoryCategory::Core, None)
        .await
        .unwrap();

    let hits = m.recall("deploy runbook", 10, None, None, None).await.unwrap();
    assert_eq!(hits[0].key, "deploy_runbook", "key match outranks content mention");
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_empty_query_returns_recent_first() {
    let m = mem();
    m.store("old", "first stored", MemoryCategory::Core, None).await.unwrap();
    tick();
    m.store("mid", "second stored", MemoryCategory::Core, None).await.unwrap();
    tick();
    m.store("new", "third stored", MemoryCategory::Core, None).await.unwrap();

    let hits = m.recall("", 10, None, None, None).await.unwrap();
    let keys: Vec<&str> = hits.iter().map(|e| e.key.as_str()).collect();
    assert_eq!(keys, vec!["new", "mid", "old"], "recency order");

    let star = m.recall("*", 10, None, None, None).await.unwrap();
    assert_eq!(star.len(), 3, "bare * behaves as empty query");
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_respects_limit() {
    let m = mem();
    // Seed one-by-one (vector-indexed table: no big batch inserts).
    for i in 0..8 {
        m.store(&format!("k{i}"), &format!("shared topic entry number {i}"), MemoryCategory::Core, None)
            .await
            .unwrap();
    }
    let hits = m.recall("shared topic", 3, None, None, None).await.unwrap();
    assert_eq!(hits.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_filters_by_session() {
    let m = mem();
    m.store("a", "meeting notes about roadmap", MemoryCategory::Conversation, Some("s1"))
        .await
        .unwrap();
    m.store("b", "meeting notes about budget", MemoryCategory::Conversation, Some("s2"))
        .await
        .unwrap();

    let hits = m.recall("meeting notes", 10, Some("s1"), None, None).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, "a");
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_time_window_since_until() {
    let m = mem();
    m.store("early", "window test entry early", MemoryCategory::Core, None).await.unwrap();
    tick();
    let boundary = chrono::Utc::now().to_rfc3339();
    tick();
    m.store("late", "window test entry late", MemoryCategory::Core, None).await.unwrap();

    let after = m
        .recall("window test", 10, None, Some(&boundary), None)
        .await
        .unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].key, "late");

    let before = m
        .recall("window test", 10, None, None, Some(&boundary))
        .await
        .unwrap();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].key, "early");
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_hybrid_semantic_match_ranks_related_first() {
    let m = mem();
    m.store("db", "distributed database survives node failure", MemoryCategory::Core, None)
        .await
        .unwrap();
    m.store("bread", "banana bread recipe with cinnamon", MemoryCategory::Core, None)
        .await
        .unwrap();

    let hits = m
        .recall("distributed database node survival", 10, None, None, None)
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].key, "db");
    let db_score = hits[0].score.unwrap();
    assert!(db_score > 0.0);
    if let Some(bread) = hits.iter().find(|e| e.key == "bread") {
        assert!(db_score > bread.score.unwrap_or(0.0));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_for_agents_allowlist_filters() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("scout").await.unwrap();
    let a2 = m.ensure_agent_uuid("builder").await.unwrap();

    m.store_with_agent("s1", "allowlist fact from scout", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();
    m.store_with_agent("b1", "allowlist fact from builder", MemoryCategory::Core, None, None, None, Some(&a2))
        .await
        .unwrap();

    let hits = m
        .recall_for_agents(&[a1.as_str()], "allowlist fact", 10, None, None, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, "s1");

    let both = m
        .recall_for_agents(&[a1.as_str(), a2.as_str()], "allowlist fact", 10, None, None, None)
        .await
        .unwrap();
    assert_eq!(both.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_namespaced_filters() {
    let m = mem();
    m.store_with_options(
        "w1",
        "namespaced entry for work",
        MemoryCategory::Core,
        None,
        StoreOptions::default().with_namespace("work"),
    )
    .await
    .unwrap();
    m.store_with_options(
        "h1",
        "namespaced entry for home",
        MemoryCategory::Core,
        None,
        StoreOptions::default().with_namespace("home"),
    )
    .await
    .unwrap();

    let hits = m
        .recall_namespaced("work", "namespaced entry", 10, None, None, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, "w1");
    assert_eq!(hits[0].namespace, "work");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_returns_all_and_filters() {
    let m = mem();
    m.store("a", "core one", MemoryCategory::Core, Some("s1")).await.unwrap();
    tick();
    m.store("b", "daily one", MemoryCategory::Daily, Some("s2")).await.unwrap();

    let all = m.list(None, None).await.unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].key, "b", "updated_at DESC");

    let core_only = m.list(Some(&MemoryCategory::Core), None).await.unwrap();
    assert_eq!(core_only.len(), 1);
    assert_eq!(core_only[0].key, "a");

    let s2_only = m.list(None, Some("s2")).await.unwrap();
    assert_eq!(s2_only.len(), 1);
    assert_eq!(s2_only[0].key, "b");
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_removes_row_and_reports() {
    let m = mem();
    m.store("gone", "to be deleted", MemoryCategory::Core, None).await.unwrap();

    assert!(m.forget("gone").await.unwrap());
    assert!(m.get("gone").await.unwrap().is_none());
    assert!(!m.forget("gone").await.unwrap(), "second delete reports false");
    assert!(!m.forget("never-existed").await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_for_agent_spares_sibling() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("alpha").await.unwrap();
    let a2 = m.ensure_agent_uuid("beta").await.unwrap();
    m.store_with_agent("k", "alpha row", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();
    m.store_with_agent("k", "beta row", MemoryCategory::Core, None, None, None, Some(&a2))
        .await
        .unwrap();

    assert!(m.forget_for_agent("k", &a1).await.unwrap());
    assert!(m.get_for_agent("k", &a1).await.unwrap().is_none());
    assert!(m.get_for_agent("k", &a2).await.unwrap().is_some(), "sibling survives");
}

#[tokio::test(flavor = "multi_thread")]
async fn purge_session_for_agent_counts() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("alpha").await.unwrap();
    let a2 = m.ensure_agent_uuid("beta").await.unwrap();
    m.store_with_agent("k1", "c1", MemoryCategory::Core, Some("sess"), None, None, Some(&a1))
        .await
        .unwrap();
    m.store_with_agent("k2", "c2", MemoryCategory::Core, Some("sess"), None, None, Some(&a1))
        .await
        .unwrap();
    m.store_with_agent("k3", "c3", MemoryCategory::Core, Some("sess"), None, None, Some(&a2))
        .await
        .unwrap();

    let purged = m.purge_session_for_agent("sess", &a1).await.unwrap();
    assert_eq!(purged, 2);
    assert_eq!(m.count().await.unwrap(), 1, "other agent's session row survives");
}

#[tokio::test(flavor = "multi_thread")]
async fn purge_agent_by_alias() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("doomed").await.unwrap();
    m.store_with_agent("k1", "c1", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();
    m.store("other", "unaffected", MemoryCategory::Core, None).await.unwrap();

    assert_eq!(m.purge_agent("doomed").await.unwrap(), 1);
    assert_eq!(m.count().await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn purge_namespace_counts() {
    let m = mem();
    m.store_with_options("a", "x", MemoryCategory::Core, None, StoreOptions::default().with_namespace("scratch"))
        .await
        .unwrap();
    m.store_with_options("b", "y", MemoryCategory::Core, None, StoreOptions::default().with_namespace("scratch"))
        .await
        .unwrap();
    m.store("keep", "z", MemoryCategory::Core, None).await.unwrap();

    assert_eq!(m.purge_namespace("scratch").await.unwrap(), 2);
    assert_eq!(m.count().await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn purge_session_counts() {
    let m = mem();
    m.store("a", "x", MemoryCategory::Core, Some("dead-sess")).await.unwrap();
    m.store("b", "y", MemoryCategory::Core, Some("dead-sess")).await.unwrap();
    m.store("c", "z", MemoryCategory::Core, Some("live-sess")).await.unwrap();

    assert_eq!(m.purge_session("dead-sess").await.unwrap(), 2);
    assert_eq!(m.count().await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn count_and_count_agent() {
    let m = mem();
    assert_eq!(m.count().await.unwrap(), 0);
    m.store("a", "x", MemoryCategory::Core, None).await.unwrap();
    assert_eq!(m.count().await.unwrap(), 1);

    m.ensure_agent_uuid("present").await.unwrap();
    assert_eq!(m.count_agent("present").await.unwrap(), 1, "alias row presence");
    assert_eq!(m.count_agent("absent").await.unwrap(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn ensure_agent_uuid_idempotent() {
    let m = mem();
    let first = m.ensure_agent_uuid("stable").await.unwrap();
    let second = m.ensure_agent_uuid("stable").await.unwrap();
    assert_eq!(first, second);
}

#[tokio::test(flavor = "multi_thread")]
async fn rename_agent_moves_alias() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("before").await.unwrap();
    m.store_with_agent("k", "carried over", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();

    let moved = m.rename_agent("before", "after").await.unwrap();
    assert_eq!(moved, 1);
    assert_eq!(m.count_agent("before").await.unwrap(), 0);
    assert_eq!(m.count_agent("after").await.unwrap(), 1);

    let exported = m.export_agent("after").await.unwrap();
    assert_eq!(exported.len(), 1);
    assert_eq!(exported[0].content, "carried over");
}

#[tokio::test(flavor = "multi_thread")]
async fn rename_agent_refuses_merge_into_nonempty() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("src").await.unwrap();
    let a2 = m.ensure_agent_uuid("dst").await.unwrap();
    m.store_with_agent("k1", "src row", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();
    m.store_with_agent("k2", "dst row", MemoryCategory::Core, None, None, None, Some(&a2))
        .await
        .unwrap();

    let err = m.rename_agent("src", "dst").await.unwrap_err();
    assert!(err.to_string().contains("refusing to merge"));
}

#[tokio::test(flavor = "multi_thread")]
async fn export_agent_ordered_by_created_at() {
    let m = mem();
    let a1 = m.ensure_agent_uuid("exporter").await.unwrap();
    m.store_with_agent("first", "one", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();
    tick();
    m.store_with_agent("second", "two", MemoryCategory::Core, None, None, None, Some(&a1))
        .await
        .unwrap();

    let exported = m.export_agent("exporter").await.unwrap();
    let keys: Vec<&str> = exported.iter().map(|e| e.key.as_str()).collect();
    assert_eq!(keys, vec!["first", "second"], "created_at ASC");
}

#[tokio::test(flavor = "multi_thread")]
async fn supersede_marks_rows() {
    let m = mem();
    m.store("belief", "the sky is green", MemoryCategory::Core, None).await.unwrap();
    let old = m.get("belief").await.unwrap().unwrap();
    m.store("belief_v2", "the sky is blue", MemoryCategory::Core, None).await.unwrap();
    let new = m.get("belief_v2").await.unwrap().unwrap();

    m.supersede(&[old.id.clone()], &new.id).await.unwrap();

    let old_after = m.get("belief").await.unwrap().unwrap();
    assert_eq!(old_after.superseded_by.as_deref(), Some(new.id.as_str()));
    assert!(m.get("belief_v2").await.unwrap().unwrap().superseded_by.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn store_with_options_persists_full_surface() {
    let m = mem();
    m.store_with_options(
        "typed",
        "tenant-scoped pinned decision",
        MemoryCategory::Core,
        None,
        StoreOptions::default()
            .with_namespace("ops")
            .with_importance(0.9)
            .with_kind(MemoryKind::Semantic(SemanticSubtype::Decision))
            .pinned(true)
            .with_tenant_id("acme"),
    )
    .await
    .unwrap();

    let hit = m.get("typed").await.unwrap().unwrap();
    assert_eq!(hit.namespace, "ops");
    assert_eq!(hit.importance, Some(0.9));
    assert_eq!(hit.kind, Some(MemoryKind::Semantic(SemanticSubtype::Decision)));
    assert!(hit.pinned);
    assert_eq!(hit.tenant_id.as_deref(), Some("acme"));
}

#[tokio::test(flavor = "multi_thread")]
async fn count_in_scope_and_stats() {
    let m = mem();
    m.store_with_options(
        "a",
        "x",
        MemoryCategory::Core,
        None,
        StoreOptions::default().with_namespace("ops").pinned(true),
    )
    .await
    .unwrap();
    m.store("b", "y", MemoryCategory::Daily, None).await.unwrap();
    let b = m.get("b").await.unwrap().unwrap();
    let a = m.get("a").await.unwrap().unwrap();
    m.supersede(&[b.id.clone()], &a.id).await.unwrap();

    assert_eq!(m.count_in_scope(Some("ops"), None).await.unwrap(), 1);
    assert_eq!(m.count_in_scope(None, Some(&MemoryCategory::Daily)).await.unwrap(), 1);
    assert_eq!(m.count_in_scope(None, None).await.unwrap(), 2);

    let stats = m.stats().await.unwrap();
    assert_eq!(stats.total_rows, 2);
    assert_eq!(stats.pinned_rows, 1);
    assert_eq!(stats.superseded_rows, 1);
    let cats: std::collections::HashMap<_, _> = stats.by_category.into_iter().collect();
    assert_eq!(cats.get("core"), Some(&1));
    assert_eq!(cats.get("daily"), Some(&1));
}

#[tokio::test(flavor = "multi_thread")]
async fn health_check_reports_true() {
    let m = mem();
    assert!(m.health_check().await);
}

#[tokio::test(flavor = "multi_thread")]
async fn session_id_sanitized_on_store() {
    let m = mem();
    m.store("k", "session sanitize check", MemoryCategory::Core, Some("slack C1.2 user one"))
        .await
        .unwrap();

    let hit = m.get("k").await.unwrap().unwrap();
    assert_eq!(hit.session_id.as_deref(), Some("slack_C1_2_user_one"));

    // Recall with the sanitized filter finds it.
    let hits = m
        .recall("session sanitize", 10, Some("slack_C1_2_user_one"), None, None)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn keyword_only_backend_still_recalls() {
    // NoopEmbedding (dims=0): no embedding column, keyword-only recall.
    let m = CockroachMemory::new(&dsn(), &unique_schema(), "memories", Some(15), None)
        .expect("connect keyword-only");
    m.store("a", "pure keyword recall entry", MemoryCategory::Core, None)
        .await
        .unwrap();

    let hits = m.recall("keyword recall", 10, None, None, None).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].key, "a");
}
