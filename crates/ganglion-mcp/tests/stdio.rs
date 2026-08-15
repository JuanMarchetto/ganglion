//! End-to-end test of the MCP stdio transport: spawns the real
//! `ganglion-mcp` binary against the real cluster and drives a full session
//! through stdin/stdout — no LLM required, just the protocol.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl Server {
    fn spawn() -> Self {
        let schema = format!("mcp_{}", uuid::Uuid::new_v4().simple());
        let dsn = std::env::var("GANGLION_TEST_DSN").unwrap_or_else(|_| {
            "postgresql://root@localhost:26257/ganglion?sslmode=disable".into()
        });
        let mut child = Command::new(env!("CARGO_BIN_EXE_ganglion-mcp"))
            .env("GANGLION_DSN", dsn)
            .env("GANGLION_SCHEMA", schema)
            .env("GANGLION_AGENT", "test-agent")
            .env("GANGLION_HMAC_KEY", "stdio-test-key")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn ganglion-mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn call(&mut self, msg: serde_json::Value) -> serde_json::Value {
        let mut line = serde_json::to_string(&msg).unwrap();
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
        let mut reply = String::new();
        self.stdout.read_line(&mut reply).expect("read reply");
        serde_json::from_str(&reply).expect("reply is JSON")
    }

    /// Call a tool and parse the text payload back into JSON.
    fn tool(&mut self, id: u64, name: &str, args: serde_json::Value) -> serde_json::Value {
        let resp = self.call(serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": { "name": name, "arguments": args }
        }));
        assert_eq!(
            resp["result"]["isError"], false,
            "tool {name} errored: {resp}"
        );
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        serde_json::from_str(text).expect("tool payload is JSON")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn full_session_over_stdio() {
    let mut s = Server::spawn();

    // Handshake.
    let init = s.call(serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2025-03-26" }
    }));
    assert_eq!(init["result"]["serverInfo"]["name"], "ganglion");
    assert_eq!(init["result"]["protocolVersion"], "2025-03-26");

    // Tool inventory.
    let list = s.call(serde_json::json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }));
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for expected in [
        "remember",
        "recall",
        "recall_asof",
        "supersede",
        "timeline",
        "verify_ledger",
        "forget",
    ] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }

    // Assert a belief, then correct it.
    let w1 = s.tool(3, "remember", serde_json::json!({
        "key": "release_day", "content": "We release on Fridays", "source": "old runbook"
    }));
    assert_eq!(w1["action"], "created");
    std::thread::sleep(std::time::Duration::from_millis(1200));
    let between = chrono_now();
    std::thread::sleep(std::time::Duration::from_millis(200));

    let w2 = s.tool(4, "remember", serde_json::json!({
        "key": "release_day", "content": "We release on Tuesdays, never Fridays", "source": "incident 12 postmortem"
    }));
    assert_eq!(w2["action"], "superseded");

    // Current recall sees only the correction.
    let hits = s.tool(5, "recall", serde_json::json!({ "query": "release day" }));
    let contents: Vec<&str> = hits
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["content"].as_str().unwrap())
        .collect();
    assert!(contents.iter().any(|c| c.contains("Tuesdays")));
    assert!(!contents.iter().any(|c| c.contains("on Fridays") && !c.contains("never")));

    // Time travel to between the two assertions.
    let then = s.tool(6, "recall_asof", serde_json::json!({
        "query": "release day", "asof": between
    }));
    let then_contents: Vec<&str> = then
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["content"].as_str().unwrap())
        .collect();
    assert_eq!(then_contents.len(), 1);
    assert!(then_contents[0].contains("We release on Fridays"));

    // Timeline shows the whole chain.
    let tl = s.tool(7, "timeline", serde_json::json!({ "key": "release_day" }));
    assert_eq!(tl.as_array().unwrap().len(), 2);

    // Ledger is clean.
    let v = s.tool(8, "verify_ledger", serde_json::json!({}));
    assert_eq!(v["clean"], true, "ledger not clean: {v}");
    let chains = v["chains"].as_array().unwrap();
    assert!(!chains.is_empty());
    assert!(chains.iter().all(|c| c["valid"] == true), "all chains valid: {v}");

    // Forget is ledgered and removes the chain.
    let f = s.tool(9, "forget", serde_json::json!({ "key": "release_day" }));
    assert_eq!(f["removed"], true);
    let tl2 = s.tool(10, "timeline", serde_json::json!({ "key": "release_day" }));
    assert_eq!(tl2.as_array().unwrap().len(), 0);
}

fn chrono_now() -> String {
    // RFC 3339 UTC without pulling chrono into dev-deps: shell out to date.
    let out = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%S.%6NZ"])
        .output()
        .expect("date");
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}
