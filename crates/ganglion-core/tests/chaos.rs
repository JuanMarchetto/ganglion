//! Kill-node chaos test: writes keep flowing while a CockroachDB node dies.
//!
//! Requires the local 3-node cluster (`deploy/local/up.sh`) **and** docker
//! privileges, so it is `#[ignore]` by default. Run explicitly:
//!
//! ```sh
//! GANGLION_CHAOS=1 cargo test -p ganglion-core --test chaos -- --ignored --nocapture
//! ```
//!
//! Invariant proven: **every write the client saw ACKed is durable** — present
//! as the current belief AND committed in the ledger — even though a node was
//! killed mid-run. (At-least-once on unACKed writes is fine; loss of an ACKed
//! write is not.)

use ganglion_core::{CockroachMemory, HashEmbedding, StoreOptions};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const KILL_NODE_ENV: &str = "GANGLION_CHAOS_NODE";
const DEFAULT_KILL_NODE: &str = "local-roach2-1";
const WRITERS: usize = 8;
const WRITES_PER_WRITER: usize = 25;

fn dsn() -> String {
    std::env::var("GANGLION_TEST_DSN")
        .unwrap_or_else(|_| "postgresql://root@localhost:26257/ganglion?sslmode=disable".into())
}

/// Restarts the killed node even if the test panics.
struct NodeGuard(String);

impl Drop for NodeGuard {
    fn drop(&mut self) {
        let _ = Command::new("docker").args(["start", &self.0]).output();
    }
}

fn docker_kill(name: &str) {
    let out = Command::new("docker")
        .args(["kill", name])
        .output()
        .expect("docker kill spawns");
    assert!(
        out.status.success(),
        "docker kill {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "chaos: kills a docker node; run with GANGLION_CHAOS=1 cargo test --test chaos -- --ignored"]
async fn acked_writes_survive_a_node_kill() {
    if std::env::var("GANGLION_CHAOS").as_deref() != Ok("1") {
        eprintln!("GANGLION_CHAOS != 1 — skipping");
        return;
    }
    let node = std::env::var(KILL_NODE_ENV).unwrap_or_else(|_| DEFAULT_KILL_NODE.into());
    let _guard = NodeGuard(node.clone());

    let schema = format!("chaos_{}", uuid::Uuid::new_v4().simple());
    let m = Arc::new(
        CockroachMemory::new(
            &dsn(),
            &schema,
            "memories",
            Some(15),
            Some(Arc::new(HashEmbedding::new(32))),
        )
        .expect("connect + init")
        .with_hmac_key(b"chaos-test-key".to_vec()),
    );

    let acked = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let failed = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let killed = Arc::new(AtomicBool::new(false));
    let acked_after_kill = Arc::new(AtomicUsize::new(0));
    let done_at_kill = Arc::new(AtomicUsize::new(usize::MAX));
    let total = WRITERS * WRITES_PER_WRITER;

    // Kill the node once ~40% of writes have been ACKed.
    let killer = {
        let done = done.clone();
        let killed = killed.clone();
        let done_at_kill = done_at_kill.clone();
        let node = node.clone();
        tokio::task::spawn_blocking(move || {
            while done.load(Ordering::Relaxed) < total * 2 / 5 {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            eprintln!("[chaos] killing {node} …");
            docker_kill(&node);
            done_at_kill.store(done.load(Ordering::Relaxed), Ordering::SeqCst);
            killed.store(true, Ordering::SeqCst);
        })
    };

    let mut writers = Vec::new();
    for w in 0..WRITERS {
        let m = m.clone();
        let acked = acked.clone();
        let failed = failed.clone();
        let done = done.clone();
        let killed = killed.clone();
        let acked_after_kill = acked_after_kill.clone();
        writers.push(tokio::spawn(async move {
            for i in 0..WRITES_PER_WRITER {
                let key = format!("chaos_{w}_{i}");
                let content = format!("observation {i} from writer {w}");
                // Sampled *before* the write: counts only writes issued while
                // the node was already dead — the ones that prove the cluster
                // kept serving, not ones that raced the kill.
                let started_after_kill = killed.load(Ordering::SeqCst);
                let res = m
                    .store_belief(&key, &content, Some("chaos"), None, StoreOptions::default(), &[])
                    .await;
                match res {
                    Ok(_) => {
                        if started_after_kill {
                            acked_after_kill.fetch_add(1, Ordering::Relaxed);
                        }
                        acked.lock().unwrap().push(key);
                    }
                    Err(e) => {
                        eprintln!("[chaos] write {key} NOT acked: {e:#}");
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                done.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for t in writers {
        t.await.unwrap();
    }
    killer.await.unwrap();

    let acked = acked.lock().unwrap().clone();
    let after_kill = acked_after_kill.load(Ordering::Relaxed);
    eprintln!(
        "[chaos] {} ACKed ({after_kill} of them issued while the node was down), \
         {} not-acked (allowed), verifying durability…",
        acked.len(),
        failed.load(Ordering::Relaxed)
    );

    // Guard against a vacuous pass: if the kill landed after the last write,
    // nothing about resilience was proven. Both must hold.
    let at_kill = done_at_kill.load(Ordering::SeqCst);
    assert!(
        at_kill < total,
        "kill landed after all {total} writes finished (at {at_kill}) — nothing was proven; \
         the writers are too fast for the killer's threshold"
    );
    assert!(
        after_kill > 0,
        "no write was issued after the node died — nothing was proven"
    );

    // Every ACKed write must be the current belief for its key.
    let mut lost = Vec::new();
    for key in &acked {
        let tl = m.belief_timeline(key, None).await.expect("timeline");
        let current = tl.iter().any(|v| v.valid_to.is_none());
        if !current {
            lost.push(key.clone());
        }
    }
    assert!(
        lost.is_empty(),
        "ACKed writes missing after node kill: {lost:?}"
    );

    // And the ledger must verify clean — every ACKed write committed.
    let v = m.verify_ledger().await.expect("verify");
    assert!(
        v.is_clean(),
        "ledger dirty after chaos: chains={:?} mismatches={:?}",
        v.chains.iter().filter(|c| !c.report.valid).count(),
        v.row_mismatches
    );
    assert!(
        v.entries_checked() >= acked.len(),
        "ledger has fewer entries ({}) than ACKed writes ({})",
        v.entries_checked(),
        acked.len()
    );

    eprintln!(
        "[chaos] OK: {}/{} ACKed writes durable, {after_kill} of them issued after \
         {node} died at write {at_kill}/{total}; ledger clean ({} entries)",
        acked.len(),
        acked.len(),
        v.entries_checked()
    );
}
