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

## Status

Hackathon build in progress (Aug 14–18, 2026). Architecture and day-by-day log in commit history.

## Prior work (declared per hackathon rules)

New work in this repo is created during the submission period. It builds on prior open source:

- [`zeroclaw`](https://github.com/zeroclaw-labs/zeroclaw) — `zeroclaw-memory` crate (MIT OR Apache-2.0): base memory engine (Postgres backend, hybrid retrieval, consolidation/decay, knowledge graph) being **ported to CockroachDB** here.
- **esclusa** (author's own prior code): HMAC append-only ledger and axum service skeleton, adapted.
- **coati** (author's own prior code): dependency-free MCP stdio server template and demo agent, adapted.

New work in this repo: the CockroachDB port (native `VECTOR`, distributed vector index with prefix columns, serializable transaction design), bitemporal facts + `recall_asof`, the single-transaction hybrid write, ledger integration, the MCP surface, self-hosted embeddings, and the AWS deployment.

## License

[Apache-2.0](LICENSE)
