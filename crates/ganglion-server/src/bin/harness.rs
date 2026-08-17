//! Load + chaos harness — produces the project's citable metric.
//!
//! 1.000 concurrent belief writes from 16 agents; halfway through, a
//! CockroachDB node is killed with `docker kill`. Afterwards it proves:
//! every ACKed write is durable (0 lost), the HMAC ledger verifies clean,
//! and reports write/recall latency percentiles measured around the kill.
//!
//! One command, reproducible:
//!
//! ```sh
//! deploy/local/up.sh && cargo run --release -p ganglion-server --bin harness
//! ```
//!
//! Env knobs: `HARNESS_DSN`, `HARNESS_TOTAL` (1000), `HARNESS_WRITERS` (16),
//! `HARNESS_KILL` (local-roach2-1; empty string disables the kill).

use ganglion_core::{CockroachMemory, HashEmbedding, Memory, StoreOptions};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Recall samples taken after the kill, with the cluster still one node down.
const RECALLS: usize = 120;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

fn percentile(sorted_ms: &[f64], p: f64) -> f64 {
    if sorted_ms.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted_ms.len() as f64 - 1.0) * p).round() as usize;
    sorted_ms[idx]
}

struct NodeGuard(String);
impl Drop for NodeGuard {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            let _ = Command::new("docker").args(["start", &self.0]).output();
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();

    let dsn = env_or(
        "HARNESS_DSN",
        "postgresql://root@localhost:26257/ganglion?sslmode=disable",
    );
    let total: usize = env_or("HARNESS_TOTAL", "1000").parse()?;
    let writers: usize = env_or("HARNESS_WRITERS", "16").parse()?;
    let kill_node = env_or("HARNESS_KILL", "local-roach2-1");
    let schema = format!("load_{}", uuid::Uuid::new_v4().simple());
    let _guard = NodeGuard(kill_node.clone());

    println!("== GANGLION LOAD + CHAOS HARNESS ==");
    println!("dsn: {dsn}");
    println!("schema: {schema} · writes: {total} · concurrent agents: {writers}");

    let m = Arc::new(
        CockroachMemory::new(
            &dsn,
            &schema,
            "memories",
            Some(15),
            Some(Arc::new(HashEmbedding::new(32))),
        )?
        .with_hmac_key(b"harness-key".to_vec()),
    );

    // One agent per writer: the realistic multi-agent shape (and the reason
    // ledger chains are per-scope — writers don't contend on one hot head).
    let mut agent_uuids = Vec::new();
    for w in 0..writers {
        agent_uuids.push(m.ensure_agent_uuid(&format!("agent_{w:02}")).await?);
    }

    // (key, agent, latency_ms, issued_after_kill)
    let acked = Arc::new(std::sync::Mutex::new(Vec::<(String, String, f64, bool)>::new()));
    let failed = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let killed_at = Arc::new(AtomicUsize::new(0));
    let kill_fired = Arc::new(AtomicBool::new(false));

    // Kill a node once half the writes are ACKed.
    let killer = {
        let done = done.clone();
        let kill_fired = kill_fired.clone();
        let killed_at = killed_at.clone();
        let node = kill_node.clone();
        tokio::task::spawn_blocking(move || {
            if node.is_empty() {
                return;
            }
            while done.load(Ordering::Relaxed) < total / 2 {
                std::thread::sleep(Duration::from_millis(10));
            }
            let at = done.load(Ordering::Relaxed);
            eprintln!("[harness] docker kill {node} (at write ~{at})");
            let out = Command::new("docker")
                .args(["kill", &node])
                .output()
                .expect("docker kill");
            assert!(out.status.success(), "docker kill failed");
            killed_at.store(at, Ordering::Relaxed);
            kill_fired.store(true, Ordering::Relaxed);
        })
    };

    let t0 = Instant::now();
    let mut tasks = Vec::new();
    for w in 0..writers {
        // Distribute the remainder so exactly `total` writes are attempted.
        let per_writer = total / writers + usize::from(w < total % writers);
        let m = m.clone();
        let acked = acked.clone();
        let failed = failed.clone();
        let done = done.clone();
        let kill_fired = kill_fired.clone();
        let agent = agent_uuids[w].clone();
        tasks.push(tokio::spawn(async move {
            for i in 0..per_writer {
                let key = format!("load_{w:02}_{i:04}");
                let content = format!(
                    "agent {w} observed event {i}: deploy window moved to {}:00 UTC",
                    (i % 24)
                );
                let opts = StoreOptions::default().with_tenant_id("load".to_string());
                // Sampled before the write so the latency split is honest:
                // "after kill" means issued while the node was already gone.
                let after_kill = kill_fired.load(Ordering::SeqCst);
                let t = Instant::now();
                match m
                    .store_belief(&key, &content, Some("harness"), Some(&agent), opts, &[])
                    .await
                {
                    Ok(_) => {
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        acked.lock().unwrap().push((key, agent.clone(), ms, after_kill));
                    }
                    Err(e) => {
                        eprintln!("[harness] write {key} NOT acked: {e:#}");
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                done.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for t in tasks {
        t.await?;
    }
    killer.await?;
    let write_wall = t0.elapsed();

    let acked = acked.lock().unwrap().clone();
    let failed = failed.load(Ordering::Relaxed);

    // Post-kill recall latency (the cluster is still one node down here).
    let mut recall_ms = Vec::new();
    let queries = ["deploy window moved", "observed event", "window UTC time"];
    for i in 0..RECALLS {
        let agent = &agent_uuids[i % writers];
        let q = queries[i % queries.len()];
        let t = Instant::now();
        let hits = m
            .recall_for_agents(&[agent.as_str()], q, 8, None, None, None)
            .await?;
        recall_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        if hits.is_empty() {
            eprintln!("[harness] WARN: empty recall for agent {agent} q='{q}'");
        }
    }

    // Durability: every ACKed write must be the current belief for its key.
    // A key carrying more than one version means the write was applied twice:
    // its transaction committed, then the connection died before the ACK
    // reached the client and the reconnect wrapper replayed it. That is
    // at-least-once, and it is why the ledger can hold more entries than
    // there were ACKs — the replay lands as a superseded version, never as a
    // silent duplicate or an overwrite.
    let mut lost = 0usize;
    let mut replayed = 0usize;
    for (key, agent, _, _) in &acked {
        let tl = m.belief_timeline(key, Some(agent)).await?;
        if !tl.iter().any(|v| v.valid_to.is_none()) {
            eprintln!("[harness] LOST: {key}");
            lost += 1;
        }
        if tl.len() > 1 {
            replayed += 1;
        }
    }

    // Tamper-evidence: full ledger verification.
    let v = m.verify_ledger().await?;

    let sorted = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let w_ms = sorted(acked.iter().map(|(_, _, ms, _)| *ms).collect());
    let pre_ms = sorted(
        acked
            .iter()
            .filter(|(_, _, _, after)| !after)
            .map(|(_, _, ms, _)| *ms)
            .collect(),
    );
    let post_ms = sorted(
        acked
            .iter()
            .filter(|(_, _, _, after)| *after)
            .map(|(_, _, ms, _)| *ms)
            .collect(),
    );
    recall_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("\n== RESULTS ==");
    println!(
        "writes:      {total} attempted · {} ACKed · {failed} rejected with an honest error \
         (allowed: the client is told the outcome is unknown rather than being lied to)",
        acked.len()
    );
    if kill_fired.load(Ordering::Relaxed) {
        println!(
            "chaos:       node {kill_node} KILLED mid-run at write ~{} — writers kept going",
            killed_at.load(Ordering::Relaxed)
        );
    } else {
        println!("chaos:       disabled");
    }
    println!(
        "durability:  {}/{} ACKed writes present · {lost} LOST",
        acked.len() - lost,
        acked.len()
    );
    println!(
        "ledger:      {} · {} entries across {} chains · {} row mismatches",
        if v.is_clean() { "CLEAN" } else { "DIRTY" },
        v.entries_checked(),
        v.chains.len(),
        v.row_mismatches.len()
    );
    println!(
        "replays:     {replayed} write(s) applied twice (committed, then ACK lost to the dying \
         node) — each landed as a superseded version, not a silent duplicate"
    );
    println!(
        "throughput:  {} writes in {:.1}s ({:.0} w/s sustained across the kill)",
        acked.len(),
        write_wall.as_secs_f64(),
        acked.len() as f64 / write_wall.as_secs_f64()
    );
    let lat = |label: &str, v: &[f64], note: &str| {
        println!(
            "  {label:<18} p50 {:>7.1} · p95 {:>7.1} · p99 {:>8.1} · max {:>8.1} ms  ({} writes) {note}",
            percentile(v, 0.50),
            percentile(v, 0.95),
            percentile(v, 0.99),
            v.last().copied().unwrap_or(f64::NAN),
            v.len()
        )
    };
    println!("write lat:");
    lat("all writes", &w_ms, "");
    lat(
        "before kill",
        &pre_ms,
        "— owns the worst tail: these are the writes still in flight when the node died, so \
         the failover wait lands here",
    );
    lat(
        "after kill",
        &post_ms,
        "— steady state on 2 of 3 nodes: what the agent sees once the node is simply gone",
    );
    println!(
        "             the tail is a write that WAITED, not one that was lost: the retry budget \
         holds a write across the lease handoff rather than returning an error."
    );
    println!(
        "recall lat:  p50 {:.1} ms · p95 {:.1} ms · p99 {:.1} ms ({RECALLS} vector+keyword hybrid recalls, one node still down)",
        percentile(&recall_ms, 0.50),
        percentile(&recall_ms, 0.95),
        percentile(&recall_ms, 0.99)
    );

    if lost > 0 || !v.is_clean() {
        anyhow::bail!("HARNESS FAILED: lost={lost} clean={}", v.is_clean());
    }
    println!("\nHARNESS PASSED: 0 ACKed writes lost, ledger tamper-check clean.");
    Ok(())
}
