//! MCP server over Ganglion memory.
//!
//! JSON-RPC 2.0 handling adapted from coati's `coati-mcp-server` (author's
//! own prior code — see ganglion-core/ATTRIBUTION.md): newline-delimited
//! stdio transport, no protocol SDK. Tools call [`CockroachMemory`] directly.

use ganglion_core::{CockroachMemory, Edge, Memory, StoreOptions};
use serde_json::{Value, json};

pub const DEFAULT_PROTOCOL_VERSION: &str = "2025-06-18";
pub const SERVER_NAME: &str = "ganglion";

pub struct McpServer {
    memory: CockroachMemory,
    /// UUID of the agent this server writes as (resolved from alias at boot).
    agent_uuid: String,
    tenant: String,
}

struct RpcError {
    code: i64,
    message: String,
}

impl RpcError {
    fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
        }
    }
    fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: message.into(),
        }
    }
}

fn required_str<'a>(args: &'a Value, field: &str) -> Result<&'a str, String> {
    args.get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("missing required string field '{field}'"))
}

impl McpServer {
    pub fn new(memory: CockroachMemory, agent_uuid: String, tenant: String) -> Self {
        Self {
            memory,
            agent_uuid,
            tenant,
        }
    }

    pub async fn handle(&self, message: Value) -> Option<Value> {
        let method = message.get("method").and_then(Value::as_str)?;
        let id = match message.get("id").cloned() {
            Some(id) if !id.is_null() => id,
            _ => return None, // notification
        };

        let outcome = match method {
            "initialize" => Ok(self.initialize_result(message.get("params"))),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(self.tools_list_result()),
            "tools/call" => self.tools_call(message.get("params")).await,
            other => Err(RpcError::method_not_found(other)),
        };

        Some(match outcome {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err(err) => json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": err.code, "message": err.message },
            }),
        })
    }

    fn initialize_result(&self, params: Option<&Value>) -> Value {
        let protocol_version = params
            .and_then(|p| p.get("protocolVersion"))
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_PROTOCOL_VERSION);
        json!({
            "protocolVersion": protocol_version,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": SERVER_NAME, "version": env!("CARGO_PKG_VERSION") },
            "instructions": "Ganglion is durable, versioned agent memory on CockroachDB. \
                `remember` asserts a belief (superseding any current version of the key), \
                `supersede` corrects an existing one, `recall` searches current beliefs, \
                `recall_asof` answers what was believed at a past instant, `timeline` \
                shows a belief's full correction history, and `verify_ledger` proves the \
                memory has not been tampered with.",
        })
    }

    fn tool_defs() -> Vec<(&'static str, &'static str, Value)> {
        vec![
            (
                "remember",
                "Assert a belief. If the key already has a current version it is superseded (history preserved), never overwritten.",
                json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string", "description": "Stable identifier for the belief, e.g. 'user_timezone'" },
                        "content": { "type": "string", "description": "The belief itself, in plain language" },
                        "source": { "type": "string", "description": "Where this came from (who said it, which doc)" },
                        "edges": {
                            "type": "array",
                            "description": "Knowledge-graph edges committed atomically with the belief",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "to_id": { "type": "string" },
                                    "relation": { "type": "string" }
                                },
                                "required": ["to_id", "relation"]
                            }
                        }
                    },
                    "required": ["key", "content"]
                }),
            ),
            (
                "recall",
                "Hybrid semantic + keyword search over current (non-superseded) beliefs.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "integer", "default": 8 }
                    },
                    "required": ["query"]
                }),
            ),
            (
                "recall_asof",
                "What was believed at a past instant? Searches the versions whose validity window contains `asof`.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "asof": { "type": "string", "description": "RFC 3339 timestamp, e.g. 2026-08-15T14:00:00Z" },
                        "limit": { "type": "integer", "default": 8 }
                    },
                    "required": ["query", "asof"]
                }),
            ),
            (
                "supersede",
                "Correct an existing belief: closes the current version and opens the new one in a single transaction. Fails if the key has no current belief.",
                json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string" },
                        "content": { "type": "string" },
                        "source": { "type": "string" }
                    },
                    "required": ["key", "content"]
                }),
            ),
            (
                "timeline",
                "Full version history of a belief: every assertion and correction with validity windows.",
                json!({
                    "type": "object",
                    "properties": { "key": { "type": "string" } },
                    "required": ["key"]
                }),
            ),
            (
                "verify_ledger",
                "Verify the tamper-evident HMAC ledger: chain integrity plus a content-hash cross-check of every memory row.",
                json!({ "type": "object", "properties": {} }),
            ),
            (
                "forget",
                "Delete a belief and its entire history for this agent (ledgered).",
                json!({
                    "type": "object",
                    "properties": { "key": { "type": "string" } },
                    "required": ["key"]
                }),
            ),
        ]
    }

    fn tools_list_result(&self) -> Value {
        let tools: Vec<Value> = Self::tool_defs()
            .into_iter()
            .map(|(name, description, schema)| {
                json!({ "name": name, "description": description, "inputSchema": schema })
            })
            .collect();
        json!({ "tools": tools })
    }

    async fn tools_call(&self, params: Option<&Value>) -> Result<Value, RpcError> {
        let params = params.ok_or_else(|| RpcError::invalid_params("missing params"))?;
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::invalid_params("missing tool name"))?;
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        // Tool-execution failures are a successful reply with isError=true,
        // per MCP, so the model can read and react to the error text.
        match self.run_tool(name, &args).await {
            Ok(value) => Ok(json!({
                "content": [{ "type": "text", "text": value.to_string() }],
                "isError": false,
            })),
            Err(err) => Ok(json!({
                "content": [{ "type": "text", "text": err }],
                "isError": true,
            })),
        }
    }

    async fn run_tool(&self, name: &str, args: &Value) -> Result<Value, String> {
        let opts = || {
            StoreOptions::default()
                .with_tenant_id(self.tenant.clone())
                .with_namespace("default")
        };
        let parse_edges = |args: &Value| -> Result<Vec<Edge>, String> {
            match args.get("edges") {
                None => Ok(Vec::new()),
                Some(v) => serde_json::from_value(v.clone())
                    .map_err(|e| format!("invalid edges: {e}")),
            }
        };

        match name {
            "remember" => {
                let key = required_str(args, "key")?;
                let content = required_str(args, "content")?;
                let source = args.get("source").and_then(Value::as_str);
                let edges = parse_edges(args)?;
                let w = self
                    .memory
                    .store_belief(key, content, source, Some(&self.agent_uuid), opts(), &edges)
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!({
                    "id": w.id,
                    "superseded_id": w.superseded_id,
                    "ledger_id": w.ledger_id,
                    "action": if w.superseded_id.is_some() { "superseded" } else { "created" },
                }))
            }
            "supersede" => {
                let key = required_str(args, "key")?;
                let content = required_str(args, "content")?;
                let source = args.get("source").and_then(Value::as_str);
                let w = self
                    .memory
                    .supersede_belief(key, content, source, Some(&self.agent_uuid), opts(), &[])
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!({ "id": w.id, "superseded_id": w.superseded_id, "ledger_id": w.ledger_id }))
            }
            "recall" => {
                let query = required_str(args, "query")?;
                let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(8) as usize;
                let hits = self
                    .memory
                    .recall(query, limit, None, None, None)
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!(hits
                    .iter()
                    .map(|e| json!({
                        "id": e.id, "key": e.key, "content": e.content,
                        "score": e.score, "timestamp": e.timestamp,
                    }))
                    .collect::<Vec<_>>()))
            }
            "recall_asof" => {
                let query = required_str(args, "query")?;
                let asof = required_str(args, "asof")?;
                let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(8) as usize;
                let hits = self
                    .memory
                    .recall_asof(query, asof, limit)
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!(hits
                    .iter()
                    .map(|e| json!({
                        "id": e.id, "key": e.key, "content": e.content, "score": e.score,
                    }))
                    .collect::<Vec<_>>()))
            }
            "timeline" => {
                let key = required_str(args, "key")?;
                let tl = self
                    .memory
                    .belief_timeline(key)
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(serde_json::to_value(tl).map_err(|e| e.to_string())?)
            }
            "verify_ledger" => {
                let v = self
                    .memory
                    .verify_ledger()
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                let clean = v.is_clean();
                let mut out = serde_json::to_value(&v).map_err(|e| e.to_string())?;
                if let Some(obj) = out.as_object_mut() {
                    obj.insert("clean".into(), json!(clean));
                }
                Ok(out)
            }
            "forget" => {
                let key = required_str(args, "key")?;
                let removed = self
                    .memory
                    .forget_for_agent(key, &self.agent_uuid)
                    .await
                    .map_err(|e| format!("{e:#}"))?;
                Ok(json!({ "removed": removed }))
            }
            other => Err(format!("unknown tool: {other}")),
        }
    }
}
