//! `ganglion-mcp` — Ganglion agent memory over the MCP `stdio` transport.
//!
//! Newline-delimited JSON-RPC 2.0 on stdin/stdout (transport shape adapted
//! from coati's `coati-mcp-server`, author's own prior code — see
//! ganglion-core/ATTRIBUTION.md).
//!
//! Configuration (environment):
//! - `GANGLION_DSN`      — CockroachDB connection string
//!                          (default: local haproxy, insecure dev cluster)
//! - `GANGLION_SCHEMA`   — storage schema (default `ganglion`)
//! - `GANGLION_AGENT`    — agent alias this server writes as (default `default`)
//! - `GANGLION_TENANT`   — tenant scope (default `default`)
//! - `GANGLION_EMBED_DIMS` — dimensions for the deterministic hash embedder
//!                          (default 32; `0` disables vector recall)
//! - `GANGLION_HMAC_KEY` — ledger signing key (dev fallback + warning if unset)

mod server;

use ganglion_core::{CockroachMemory, HashEmbedding, Memory};
use server::McpServer;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logs must go to stderr: stdout is the MCP transport.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::INFO)
        .init();

    let dsn = env_or(
        "GANGLION_DSN",
        "postgresql://root@localhost:26257/ganglion?sslmode=disable",
    );
    let schema = env_or("GANGLION_SCHEMA", "ganglion");
    let agent_alias = env_or("GANGLION_AGENT", "default");
    let tenant = env_or("GANGLION_TENANT", "default");
    let dims: usize = env_or("GANGLION_EMBED_DIMS", "32").parse().unwrap_or(32);

    let embedder = if dims > 0 {
        Some(Arc::new(HashEmbedding::new(dims)) as Arc<dyn ganglion_core::EmbeddingProvider>)
    } else {
        None
    };

    let memory = CockroachMemory::new(&dsn, &schema, "memories", Some(15), embedder)?;
    let agent_uuid = memory.ensure_agent_uuid(&agent_alias).await?;
    tracing::info!(schema, agent_alias, %agent_uuid, "ganglion-mcp ready");

    let server = McpServer::new(memory, agent_uuid, tenant);

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(message) => server.handle(message).await,
            Err(err) => Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": serde_json::Value::Null,
                "error": { "code": -32700, "message": format!("parse error: {err}") },
            })),
        };
        if let Some(response) = response {
            let mut bytes = serde_json::to_vec(&response)?;
            bytes.push(b'\n');
            stdout.write_all(&bytes).await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}
