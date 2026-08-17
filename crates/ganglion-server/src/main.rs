//! Ganglion demo server: JSON API + embedded single-page UI over the memory,
//! plus a token-gated chaos endpoint that kills a CockroachDB node live.
//!
//! Auth-by-header pattern adapted from esclusa's `gate/src/http.rs` (author's
//! own prior code — see ganglion-core/ATTRIBUTION.md).
//!
//! The core memory is a sync client behind its own OS-thread hop
//! (`run_on_os_thread`) with a reconnecting pool inside — so axum handlers
//! simply `.await` it; no extra spawn_blocking layer is needed here.

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ganglion_core::{CockroachMemory, HashEmbedding, Memory, StoreOptions};
use serde_json::{Value, json};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const UI: &str = include_str!("ui.html");
/// Only these containers may ever be touched by the chaos endpoints.
const CHAOS_ALLOWLIST: [&str; 3] = ["local-roach1-1", "local-roach2-1", "local-roach3-1"];

struct App {
    memory: CockroachMemory,
    agent_uuid: String,
    admin_token: String,
    chaos_enabled: bool,
    started: Instant,
    remembers: AtomicU64,
    recalls: AtomicU64,
    lat: std::sync::Mutex<LatRing>,
}

/// Fixed-size ring of recent op latencies (ms) for the live metrics panel.
struct LatRing {
    remember: Vec<f64>,
    recall: Vec<f64>,
}

impl LatRing {
    fn push(v: &mut Vec<f64>, ms: f64) {
        if v.len() >= 512 {
            v.remove(0);
        }
        v.push(ms);
    }
    fn pct(v: &[f64], p: f64) -> Option<f64> {
        if v.is_empty() {
            return None;
        }
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(s[((s.len() as f64 - 1.0) * p).round() as usize])
    }
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.into())
}

fn err(status: StatusCode, msg: impl std::fmt::Display) -> Response {
    (status, Json(json!({ "error": msg.to_string() }))).into_response()
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();

    let dsn = env_or(
        "GANGLION_DSN",
        "postgresql://root@localhost:26257/ganglion?sslmode=disable",
    );
    let schema = env_or("GANGLION_SCHEMA", "demo");
    let agent_alias = env_or("GANGLION_AGENT_ALIAS", "demo-agent");
    let admin_token = env_or("GANGLION_ADMIN_TOKEN", "");
    let chaos_enabled = env_or("GANGLION_ENABLE_CHAOS", "") == "1";
    let port: u16 = env_or("PORT", "3000").parse()?;

    let memory = CockroachMemory::new(
        &dsn,
        &schema,
        "memories",
        Some(20),
        Some(Arc::new(HashEmbedding::new(32))),
    )?;
    let agent_uuid = memory.ensure_agent_uuid(&agent_alias).await?;

    if chaos_enabled && admin_token.is_empty() {
        anyhow::bail!("GANGLION_ENABLE_CHAOS=1 requires GANGLION_ADMIN_TOKEN to be set");
    }

    let app = Arc::new(App {
        memory,
        agent_uuid,
        admin_token,
        chaos_enabled,
        started: Instant::now(),
        remembers: AtomicU64::new(0),
        recalls: AtomicU64::new(0),
        lat: std::sync::Mutex::new(LatRing {
            remember: Vec::new(),
            recall: Vec::new(),
        }),
    });

    let router = Router::new()
        .route("/", get(|| async { Html(UI) }))
        .route("/healthz", get(|| async { "ok" }))
        .route("/api/remember", post(api_remember))
        .route("/api/recall", post(api_recall))
        .route("/api/recall_asof", post(api_recall_asof))
        .route("/api/supersede", post(api_supersede))
        .route("/api/timeline/{key}", get(api_timeline))
        .route("/api/verify", get(api_verify))
        .route("/api/metrics", get(api_metrics))
        .route("/api/nodes", get(api_nodes))
        .route("/api/chaos/kill", post(api_chaos_kill))
        .route("/api/chaos/revive", post(api_chaos_revive))
        .with_state(app);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("ganglion-server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

fn req_str<'a>(body: &'a Value, field: &str) -> Result<&'a str, Response> {
    body.get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, format!("missing field '{field}'")))
}

fn limit_of(body: &Value) -> usize {
    body.get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(8)
        .clamp(1, 100) as usize
}

async fn api_remember(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let key = match req_str(&body, "key") {
        Ok(k) => k.to_string(),
        Err(e) => return e,
    };
    let content = match req_str(&body, "content") {
        Ok(c) => c.to_string(),
        Err(e) => return e,
    };
    let source = body.get("source").and_then(Value::as_str).map(String::from);

    let t = Instant::now();
    let res = app
        .memory
        .store_belief(
            &key,
            &content,
            source.as_deref(),
            Some(&app.agent_uuid),
            StoreOptions::default(),
            &[],
        )
        .await;
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    app.remembers.fetch_add(1, Ordering::Relaxed);
    LatRing::push(&mut app.lat.lock().unwrap().remember, ms);

    match res {
        Ok(w) => Json(json!({
            "id": w.id,
            "superseded_id": w.superseded_id,
            "ledger_id": w.ledger_id,
            "latency_ms": ms,
            "action": if w.superseded_id.is_some() { "superseded" } else { "created" },
        }))
        .into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn api_supersede(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let key = match req_str(&body, "key") {
        Ok(k) => k.to_string(),
        Err(e) => return e,
    };
    let content = match req_str(&body, "content") {
        Ok(c) => c.to_string(),
        Err(e) => return e,
    };
    let source = body.get("source").and_then(Value::as_str).map(String::from);
    match app
        .memory
        .supersede_belief(
            &key,
            &content,
            source.as_deref(),
            Some(&app.agent_uuid),
            StoreOptions::default(),
            &[],
        )
        .await
    {
        Ok(w) => Json(json!({ "id": w.id, "superseded_id": w.superseded_id })).into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

async fn api_recall(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let query = match req_str(&body, "query") {
        Ok(q) => q.to_string(),
        Err(e) => return e,
    };
    let t = Instant::now();
    let res = app
        .memory
        .recall_for_agents(
            &[app.agent_uuid.as_str()],
            &query,
            limit_of(&body),
            None,
            None,
            None,
        )
        .await;
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    app.recalls.fetch_add(1, Ordering::Relaxed);
    LatRing::push(&mut app.lat.lock().unwrap().recall, ms);

    match res {
        Ok(hits) => Json(json!({
            "latency_ms": ms,
            "hits": hits.iter().map(|h| json!({
                "id": h.id, "key": h.key, "content": h.content, "score": h.score,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn api_recall_asof(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let query = match req_str(&body, "query") {
        Ok(q) => q.to_string(),
        Err(e) => return e,
    };
    let asof = match req_str(&body, "asof") {
        Ok(a) => a.to_string(),
        Err(e) => return e,
    };
    match app
        .memory
        .recall_asof(&query, &asof, limit_of(&body), Some(&app.agent_uuid))
        .await
    {
        Ok(hits) => Json(json!({
            "asof": asof,
            "hits": hits.iter().map(|h| json!({
                "id": h.id, "key": h.key, "content": h.content, "score": h.score,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

async fn api_timeline(State(app): State<Arc<App>>, Path(key): Path<String>) -> Response {
    match app.memory.belief_timeline(&key, Some(&app.agent_uuid)).await {
        Ok(tl) => Json(json!({ "key": key, "versions": tl })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn api_verify(State(app): State<Arc<App>>) -> Response {
    match app.memory.verify_ledger().await {
        Ok(v) => {
            let clean = v.is_clean();
            let entries = v.entries_checked();
            Json(json!({
                "clean": clean,
                "entries_checked": entries,
                "chains": v.chains,
                "row_mismatches": v.row_mismatches,
                "rows_checked": v.rows_checked,
            }))
            .into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn api_metrics(State(app): State<Arc<App>>) -> Response {
    let lat = app.lat.lock().unwrap();
    Json(json!({
        "uptime_secs": app.started.elapsed().as_secs(),
        "remembers": app.remembers.load(Ordering::Relaxed),
        "recalls": app.recalls.load(Ordering::Relaxed),
        "remember_p50_ms": LatRing::pct(&lat.remember, 0.50),
        "remember_p99_ms": LatRing::pct(&lat.remember, 0.99),
        "recall_p50_ms": LatRing::pct(&lat.recall, 0.50),
        "recall_p99_ms": LatRing::pct(&lat.recall, 0.99),
    }))
    .into_response()
}

async fn api_nodes(State(app): State<Arc<App>>) -> Response {
    // Two views merged: what docker says (instant, survives SQL being down)
    // and what the database's own gossip says.
    let docker = tokio::task::spawn_blocking(|| {
        CHAOS_ALLOWLIST
            .iter()
            .map(|name| {
                let out = Command::new("docker")
                    .args(["inspect", "-f", "{{.State.Status}}", name])
                    .output();
                let status = out
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .unwrap_or_else(|| "unknown".into());
                json!({ "container": name, "status": status })
            })
            .collect::<Vec<_>>()
    })
    .await
    .unwrap_or_default();

    let sql = app.memory.cluster_nodes().await.ok().map(|nodes| {
        nodes
            .into_iter()
            .map(|(id, addr, live)| json!({ "node_id": id, "address": addr, "is_live": live }))
            .collect::<Vec<_>>()
    });

    Json(json!({ "docker": docker, "sql_gossip": sql })).into_response()
}

/// Both chaos endpoints: require the feature flag AND the admin token AND an
/// allowlisted container name — this is deliberately not a general remote
/// docker executor.
fn chaos_gate<'a>(app: &App, headers: &HeaderMap, body: &'a Value) -> Result<&'a str, Response> {
    if !app.chaos_enabled {
        return Err(err(
            StatusCode::FORBIDDEN,
            "chaos endpoints disabled (GANGLION_ENABLE_CHAOS)",
        ));
    }
    match bearer(headers) {
        Some(t) if !app.admin_token.is_empty() && t == app.admin_token => {}
        _ => return Err(err(StatusCode::UNAUTHORIZED, "bad or missing bearer token")),
    }
    let node = body
        .get("node")
        .and_then(Value::as_str)
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "missing field 'node'"))?;
    if !CHAOS_ALLOWLIST.contains(&node) {
        return Err(err(StatusCode::BAD_REQUEST, "node not in allowlist"));
    }
    Ok(node)
}

async fn chaos_docker(cmd: &'static str, node: String) -> Response {
    let out =
        tokio::task::spawn_blocking(move || Command::new("docker").args([cmd, &node]).output())
            .await;
    match out {
        Ok(Ok(o)) if o.status.success() => {
            Json(json!({ "ok": true, "action": cmd })).into_response()
        }
        Ok(Ok(o)) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            String::from_utf8_lossy(&o.stderr),
        ),
        _ => err(StatusCode::INTERNAL_SERVER_ERROR, "docker invocation failed"),
    }
}

async fn api_chaos_kill(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    match chaos_gate(&app, &headers, &body) {
        Ok(node) => chaos_docker("kill", node.to_string()).await,
        Err(e) => e,
    }
}

async fn api_chaos_revive(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    match chaos_gate(&app, &headers, &body) {
        Ok(node) => chaos_docker("start", node.to_string()).await,
        Err(e) => e,
    }
}
