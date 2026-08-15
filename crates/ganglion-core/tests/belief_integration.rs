//! Integration tests for the belief surface: bitemporal versioning,
//! `recall_asof`, the tamper-evident ledger (both tamper axes), hybrid-write
//! atomicity, and concurrent supersedes under serializable isolation.
//!
//! Requires a running cluster (see `deploy/local/up.sh`); DSN from
//! `GANGLION_TEST_DSN`. Each test gets its own schema.

use ganglion_core::{
    CockroachMemory, Edge, HashEmbedding, Memory, MemoryCategory, StoreOptions,
};
use std::sync::Arc;
use std::time::Duration;

fn dsn() -> String {
    std::env::var("GANGLION_TEST_DSN")
        .unwrap_or_else(|_| "postgresql://root@localhost:26257/ganglion?sslmode=disable".into())
}

fn unique_schema() -> String {
    format!("b_{}", uuid::Uuid::new_v4().simple())
}

fn mem() -> CockroachMemory {
    CockroachMemory::new(
        &dsn(),
        &unique_schema(),
        "memories",
        Some(15),
        Some(Arc::new(HashEmbedding::new(32))),
    )
    .expect("connect + init schema")
    .with_hmac_key(b"belief-test-key".to_vec())
}

fn tick() {
    std::thread::sleep(Duration::from_millis(30));
}

/// Run raw SQL against the cluster on a blocking thread (the sync
/// `postgres::Client` cannot be created inside the tokio runtime).
async fn with_raw_client<T: Send + 'static>(
    f: impl FnOnce(&mut postgres::Client) -> T + Send + 'static,
) -> T {
    tokio::task::spawn_blocking(move || {
        let mut c = postgres::Client::connect(&dsn(), postgres::NoTls).expect("raw connect");
        f(&mut c)
    })
    .await
    .expect("blocking task")
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

#[tokio::test(flavor = "multi_thread")]
async fn supersede_builds_a_chain_and_recall_sees_only_current() {
    let m = mem();
    let w1 = m
        .store_belief("favorite_db", "User prefers Postgres", Some("onboarding"), None, StoreOptions::default(), &[])
        .await
        .unwrap();
    assert!(w1.superseded_id.is_none());
    tick();

    let w2 = m
        .store_belief("favorite_db", "User prefers CockroachDB", Some("user-correction"), None, StoreOptions::default(), &[])
        .await
        .unwrap();
    assert_eq!(w2.superseded_id.as_deref(), Some(w1.id.as_str()));

    // Current view: exactly the new belief.
    let hit = m.get("favorite_db").await.unwrap().expect("current row");
    assert_eq!(hit.id, w2.id);
    assert_eq!(hit.content, "User prefers CockroachDB");

    // recall() must not surface the superseded version.
    let hits = m.recall("prefers", 10, None, None, None).await.unwrap();
    assert!(hits.iter().any(|e| e.id == w2.id));
    assert!(!hits.iter().any(|e| e.id == w1.id), "superseded row leaked into recall");

    // Timeline: both versions, closed→open, correctly linked.
    let tl = m.belief_timeline("favorite_db").await.unwrap();
    assert_eq!(tl.len(), 2);
    assert_eq!(tl[0].id, w1.id);
    assert_eq!(tl[0].superseded_by.as_deref(), Some(w2.id.as_str()));
    assert!(tl[0].valid_to.is_some(), "old version must be closed");
    assert_eq!(tl[1].id, w2.id);
    assert!(tl[1].valid_to.is_none(), "new version must be open");
}

#[tokio::test(flavor = "multi_thread")]
async fn supersede_belief_requires_an_existing_current_version() {
    let m = mem();
    let err = m
        .supersede_belief("never_asserted", "content", None, None, StoreOptions::default(), &[])
        .await
        .expect_err("supersede on missing key must fail");
    assert!(err.to_string().contains("no current belief"));

    // And the failed strict supersede must not have written anything.
    assert!(m.get("never_asserted").await.unwrap().is_none());
    assert_eq!(m.ledger_entries().await.unwrap().len(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_asof_returns_what_was_believed_then() {
    let m = mem();
    m.store_belief("deploy_region", "We deploy in us-east-1 only", Some("runbook"), None, StoreOptions::default(), &[])
        .await
        .unwrap();
    tick();
    let between = now_rfc3339();
    tick();
    m.store_belief("deploy_region", "We deploy multi-region eu-west-1 and us-east-1", Some("migration"), None, StoreOptions::default(), &[])
        .await
        .unwrap();

    // As of `between`, the old belief was current.
    let then = m.recall_asof("deploy region", &between, 10).await.unwrap();
    assert_eq!(then.len(), 1, "exactly one version valid at t");
    assert!(then[0].content.contains("us-east-1 only"));

    // As of now, the new one.
    let now = m.recall_asof("deploy region", &now_rfc3339(), 10).await.unwrap();
    assert_eq!(now.len(), 1);
    assert!(now[0].content.contains("multi-region"));

    // Before the first assertion: nothing.
    let before = m
        .recall_asof("deploy region", "2020-01-01T00:00:00Z", 10)
        .await
        .unwrap();
    assert!(before.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn asof_system_time_agrees_with_applicative_window() {
    let m = mem();
    m.store_belief("owner", "Service owned by team A", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();
    // AS OF SYSTEM TIME reads schema history: the timestamp must postdate
    // schema creation, so give the cluster a moment and use a fresh instant.
    std::thread::sleep(Duration::from_millis(1500));
    let t = now_rfc3339();
    std::thread::sleep(Duration::from_millis(200));
    m.store_belief("owner", "Service owned by team B", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();

    // MVCC axis: at t the *current* row said team A.
    let mvcc = m
        .get_asof_system_time("owner", &t)
        .await
        .unwrap()
        .expect("row existed at t");
    assert!(mvcc.content.contains("team A"));

    // Applicative axis must agree.
    let app = m.recall_asof("owner service", &t, 10).await.unwrap();
    assert_eq!(app.len(), 1);
    assert!(app[0].content.contains("team A"));
}

#[tokio::test(flavor = "multi_thread")]
async fn ledger_chain_is_valid_after_normal_operations() {
    let m = mem();
    m.store("k1", "v1", MemoryCategory::Core, None).await.unwrap();
    m.store_belief("k2", "belief", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();
    m.store_belief("k2", "corrected belief", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();
    m.forget("k1").await.unwrap();

    let entries = m.ledger_entries().await.unwrap();
    assert_eq!(entries.len(), 4, "store, store, supersede, forget");
    assert_eq!(entries[0].kind, "store");
    assert_eq!(entries[2].kind, "supersede");
    assert_eq!(entries[3].kind, "forget");

    let v = m.verify_ledger().await.unwrap();
    assert!(v.chain.valid, "chain must verify: {:?}", v.chain);
    assert!(v.row_mismatches.is_empty(), "no tampering yet: {:?}", v.row_mismatches);
    // k1 was forgotten (row deleted); k2 has two versions (closed + current),
    // both cross-checked against their latest ledger hashes.
    assert_eq!(v.rows_checked, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn tampering_a_memory_row_is_detected_by_verify() {
    let m = mem();
    let w = m
        .store_belief("api_budget", "Monthly API budget is $500", Some("finance"), None, StoreOptions::default(), &[])
        .await
        .unwrap();

    // Attacker with SQL access rewrites the belief in place. The row id is
    // unique across schemas, so sweep every test schema until it lands.
    let target_id = w.id.clone();
    let hit = with_raw_client(move |client| {
        let schemas = client
            .query(
                "SELECT table_schema FROM information_schema.tables WHERE table_name = 'memories'",
                &[],
            )
            .unwrap();
        let mut hit = false;
        for row in schemas {
            let schema: String = row.get(0);
            let n = client
                .execute(
                    &format!(
                        "UPDATE \"{schema}\".memories SET content = 'Monthly API budget is $50000' WHERE id = $1"
                    ),
                    &[&target_id],
                )
                .unwrap_or(0);
            if n > 0 {
                hit = true;
            }
        }
        hit
    })
    .await;
    assert!(hit, "tamper UPDATE must land");

    let v = m.verify_ledger().await.unwrap();
    assert!(v.chain.valid, "ledger itself untouched");
    assert_eq!(v.row_mismatches.len(), 1, "the tampered row is flagged");
    assert_eq!(v.row_mismatches[0].id, w.id);
    assert_eq!(v.row_mismatches[0].key, "api_budget");
}

#[tokio::test(flavor = "multi_thread")]
async fn tampering_the_ledger_breaks_the_chain_at_that_id() {
    let m = mem();
    m.store_belief("a", "one", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();
    m.store_belief("b", "two", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();
    m.store_belief("c", "three", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();

    // Rewrite ledger entry 2's params directly. The `target IN (... key='b')`
    // guard pins the sweep to THIS test's schema.
    let hit = with_raw_client(|client| {
        let schemas = client
            .query(
                "SELECT table_schema FROM information_schema.tables WHERE table_name = 'memory_ledger'",
                &[],
            )
            .unwrap();
        let mut hit = false;
        for row in schemas {
            let schema: String = row.get(0);
            let n = client
                .execute(
                    &format!(
                        "UPDATE \"{schema}\".memory_ledger SET params = '{{\"key\":\"b\",\"content_sha256\":\"forged\"}}'::JSONB WHERE id = 2 AND target IN (SELECT id FROM \"{schema}\".memories WHERE key = 'b')"
                    ),
                    &[],
                )
                .unwrap_or(0);
            if n > 0 {
                hit = true;
            }
        }
        hit
    })
    .await;
    assert!(hit, "ledger tamper must land");

    let v = m.verify_ledger().await.unwrap();
    assert!(!v.chain.valid);
    assert_eq!(v.chain.broken_at, Some(2));
}

#[tokio::test(flavor = "multi_thread")]
async fn hybrid_write_is_atomic_a_bad_edge_rolls_back_everything() {
    let m = mem();
    let head_before = m.ledger_entries().await.unwrap().len();

    // Edge to a nonexistent row id: FK violation fires after the belief row
    // and the ledger entry were written inside the txn — everything must go.
    let err = m
        .store_belief(
            "poisoned",
            "this belief must not survive",
            None,
            None,
            StoreOptions::default(),
            &[Edge { to_id: "no-such-row".into(), relation: "supports".into() }],
        )
        .await
        .expect_err("dangling edge must abort the write");
    let msg = format!("{err:#}");
    assert!(msg.contains("foreign key") || msg.contains("violates"), "unexpected error: {msg}");

    assert!(m.get("poisoned").await.unwrap().is_none(), "belief row leaked");
    assert_eq!(
        m.ledger_entries().await.unwrap().len(),
        head_before,
        "ledger entry leaked from an aborted transaction"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn edges_commit_with_the_belief_and_cascade_on_forget() {
    let m = mem();
    let anchor = m
        .store_belief("service_map", "Payments depends on Auth", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();
    let w = m
        .store_belief(
            "incident_42",
            "Outage caused by Auth cert expiry",
            Some("postmortem"),
            None,
            StoreOptions::default(),
            &[Edge { to_id: anchor.id.clone(), relation: "explains".into() }],
        )
        .await
        .unwrap();

    // The edge is queryable via raw SQL in the row's schema.
    let (from_id, to_id) = (w.id.clone(), anchor.id.clone());
    let found = with_raw_client(move |client| {
        let schemas = client
            .query(
                "SELECT table_schema FROM information_schema.tables WHERE table_name = 'memory_edges'",
                &[],
            )
            .unwrap();
        let mut found = 0i64;
        for row in schemas {
            let schema: String = row.get(0);
            if let Ok(r) = client.query_one(
                &format!(
                    "SELECT count(*) FROM \"{schema}\".memory_edges WHERE from_id = $1 AND to_id = $2 AND relation = 'explains'"
                ),
                &[&from_id, &to_id],
            ) {
                found += r.get::<_, i64>(0);
            }
        }
        found
    })
    .await;
    assert_eq!(found, 1, "edge committed with the belief");

    // forget must cascade edges (no FK error).
    assert!(m.forget("incident_42").await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_supersedes_of_the_same_key_serialize_via_retry() {
    let m = Arc::new(mem());
    m.store_belief("hot_key", "v0", None, None, StoreOptions::default(), &[])
        .await
        .unwrap();

    // Two writers correct the same belief at once. Serializable isolation
    // surfaces 40001 to one of them; with_txn_retry replays it. Both must
    // succeed and the chain must be linear (no lost update, no fork).
    let m1 = m.clone();
    let m2 = m.clone();
    let (r1, r2) = tokio::join!(
        m1.store_belief("hot_key", "v1 from writer A", Some("A"), None, StoreOptions::default(), &[]),
        m2.store_belief("hot_key", "v1 from writer B", Some("B"), None, StoreOptions::default(), &[]),
    );
    r1.unwrap();
    r2.unwrap();

    let tl = m.belief_timeline("hot_key").await.unwrap();
    assert_eq!(tl.len(), 3, "v0 + both corrections");
    let open: Vec<_> = tl.iter().filter(|v| v.valid_to.is_none()).collect();
    assert_eq!(open.len(), 1, "exactly one current version after the race");
    // Linear chain: every closed version names its successor.
    for v in tl.iter().filter(|v| v.valid_to.is_some()) {
        assert!(v.superseded_by.is_some(), "closed version without successor");
    }

    let v = m.verify_ledger().await.unwrap();
    assert!(v.chain.valid, "ledger stayed a valid chain through the race");
    assert!(v.row_mismatches.is_empty());
}
