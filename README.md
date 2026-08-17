# Ganglion

**Unkillable agent memory on CockroachDB.**

A cockroach survives losing its head because its nervous system is distributed across ganglia. Ganglion gives AI agents the same property: distributed, transactional memory on CockroachDB that keeps remembering while you kill nodes mid-conversation — with time-travel over the agent's beliefs and a tamper-evident audit trail.

Built for the [CockroachDB × AWS Hackathon — Build with Agentic Memory](https://cockroachdb-ai.devpost.com/) (August 2026).

## What it does

Every memory write commits **atomically, in one serializable transaction**:

1. the **embedding** — native `VECTOR` column + distributed vector index (C-SPANN) with tenant/agent prefix columns, so multi-tenant isolation lives in the index itself;
2. the **fact**, with a temporal validity window (`valid_from` / `valid_to` / `superseded_by`) — correcting a belief supersedes it, never erases it;
3. the **knowledge-graph edge** relating it to other facts;
4. an entry in an **HMAC-chained append-only ledger** — tamper a row by hand and `/verify` catches it.

No memory framework (Mem0, Zep, Letta, Cognee) can offer that atomicity, because they split vector store and knowledge store into separate systems.

- `recall_asof(t)` — *what did the agent believe on Tuesday at 3pm, and who corrected it?* Applicative bitemporality combined with CockroachDB's `AS OF SYSTEM TIME`.
- Exposed via **MCP** (`remember`, `recall`, `recall_asof`, `supersede`, `verify_ledger`) so any MCP-capable agent can plug in.
- Survives node failure live: 3-node CockroachDB cluster behind haproxy; kill a node mid-conversation, zero memories lost.

## Run it

Needs Docker and a Rust toolchain. Nothing else — the UI is compiled into the
binary, so there is no build step and no node_modules.

```sh
deploy/local/up.sh                    # 3-node CockroachDB cluster + haproxy
cargo run --release -p ganglion-server
```

Open <http://localhost:3000>. First boot against a new schema creates the
tables and builds the vector index before the server answers anything —
measured between 40s and a couple of minutes, since the DDL waits on a
background index job. Restarts against an existing schema are immediate.

To enable the KILL NODE button, give it a token and turn chaos on:

```sh
GANGLION_ENABLE_CHAOS=1 GANGLION_ADMIN_TOKEN=$(openssl rand -hex 24) \
  cargo run --release -p ganglion-server
```

Paste the token into the UI's *admin token* field. Without both the env flag
and a matching token the chaos endpoints refuse every request, and they only
ever act on a fixed allowlist of node containers compiled into the binary.

| Variable | Default | Meaning |
|---|---|---|
| `GANGLION_DSN` | local haproxy | CockroachDB connection string |
| `GANGLION_SCHEMA` | `demo` | schema to create/use |
| `GANGLION_HMAC_KEY` | dev key (warns) | signs the ledger — set it for real use |
| `GANGLION_ENABLE_CHAOS` | off | enables `/api/chaos/*` |
| `GANGLION_ADMIN_TOKEN` | none | bearer token for `/api/chaos/*` |
| `GANGLION_POOL_SIZE` | 8 | SQL connections |
| `PORT` | 3000 | HTTP port |

## Measured, not claimed

"Survives node failure" is a testable statement, so it is tested. The harness
writes 1000 beliefs from 16 concurrent agents, kills a CockroachDB node with
`docker kill` halfway through, and then checks that every write the client was
told had succeeded is still there.

```sh
deploy/local/up.sh                                          # 3 nodes + haproxy
cargo run --release -p ganglion-server --bin harness        # writes, kills, verifies
```

Real output from one run on the local 3-node cluster:

```
writes:      1000 attempted · 993 ACKed · 7 rejected with an honest error
chaos:       node local-roach2-1 KILLED mid-run at write ~500 — writers kept going
durability:  993/993 ACKed writes present · 0 LOST
ledger:      CLEAN · 995 entries across 16 chains · 0 row mismatches
throughput:  993 writes in 17.4s (57 w/s sustained across the kill)
write lat:
  after kill         p50   125.2 · p95   319.4 · p99    577.3 ms  (474 writes)
recall lat:  p50 13.0 ms · p95 26.2 ms · p99 228.6 ms (one node still down)
```

Throughput and latency move run to run (57–87 w/s depending on how warm the
ranges are). The two numbers that do not move are the ones the design is
about: **0 ACKed writes lost, ledger verifies clean.**

Read the numbers honestly:

- **"Rejected with an honest error"** are writes CockroachDB reported as
  *ambiguous* — it could not confirm the outcome to the client. Some of them
  did commit. Being told "unknown" is the correct answer; the failure mode
  worth preventing is a write reported as saved that is not.
- **The tail is a write that waited, not one that was lost.** A write caught by
  the lease handoff sits in the retry budget until a surviving node takes over.
  The alternative to a slow write here is a failed one.
- **A replay is not a duplicate.** If a write commits and the ACK dies with the
  node, the reconnect replays it and it lands as a *superseded version* — which
  is why the ledger can hold more entries than there were ACKs, and why the
  timeline still shows exactly one current belief per key.

The same invariant is asserted as a test, so it can fail in CI rather than in a
demo:

```sh
GANGLION_CHAOS=1 cargo test -p ganglion-core --test chaos -- --ignored --nocapture
```

Three consecutive runs: 200/200, 199/199, 200/200 ACKed writes durable, ledger
clean, with ~112 writes per run issued *while the node was already dead*. The
test asserts that last part, so it cannot pass by killing the node after the
writes finish.

## Status

Hackathon build in progress (Aug 14–18, 2026). Architecture and day-by-day log in commit history.

Local deployment (3 nodes + haproxy) is `deploy/local/`; the AWS deployment
skeleton is `deploy/aws/` and has not been applied to a live account yet.

## Prior work (declared per hackathon rules)

New work in this repo is created during the submission period. It builds on prior open source:

- [`zeroclaw`](https://github.com/zeroclaw-labs/zeroclaw) — `zeroclaw-memory` crate (MIT OR Apache-2.0): base memory engine (Postgres backend, hybrid retrieval, consolidation/decay, knowledge graph) being **ported to CockroachDB** here.
- **esclusa** (author's own prior code): HMAC append-only ledger and axum service skeleton, adapted.
- **coati** (author's own prior code): dependency-free MCP stdio server template and demo agent, adapted.

New work in this repo: the CockroachDB port (native `VECTOR`, distributed vector index with prefix columns, serializable transaction design), bitemporal facts + `recall_asof`, the single-transaction hybrid write, ledger integration, the MCP surface, self-hosted embeddings, and the AWS deployment.

## License

[Apache-2.0](LICENSE)
