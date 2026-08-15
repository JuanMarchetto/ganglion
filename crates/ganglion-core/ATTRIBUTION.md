# Attribution

Parts of this crate are vendored from **zeroclaw**
(https://github.com/zeroclaw-labs/zeroclaw), licensed **MIT OR Apache-2.0**,
during the CockroachDB × AWS Agentic Memory Hackathon (Aug 2026). Per the
hackathon rules, pre-existing code is declared here; everything listed as
"new" was written during the submission period.

| File | Origin | Changes |
|---|---|---|
| `src/vector.rs` | `zeroclaw-memory/src/vector.rs` | verbatim |
| `src/chunker.rs` | `zeroclaw-memory/src/chunker.rs` | verbatim |
| `src/traits.rs` | `zeroclaw-api/src/memory_traits.rs` + `session_keys.rs` | trimmed: removed `Attributable` supertrait, `MemoryStrategy`, provider coupling; merged `sanitize_session_key` |
| `src/embeddings.rs` | `zeroclaw-memory/src/embeddings.rs` | zeroclaw log/config plumbing → `tracing` + plain reqwest; **new** `HashEmbedding` deterministic provider |
| `src/cockroach.rs` | ported from `zeroclaw-memory/src/postgres.rs` | **new work**: CockroachDB backend — native `VECTOR`, C-SPANN vector index with `(tenant_id, agent_id)` prefix columns, hybrid vector+FTS recall (`ts_rank`; `ts_rank_cd` is unimplemented in CockroachDB), fresh final schema without legacy migrations, full `StoreOptions` (kind/pinned/tenant_id) persistence, `supersede`/`stats`/purge overrides |

Not vendored (deliberately): `sqlite.rs` (reference behavior only — its
tests informed `tests/cockroach_integration.rs`), `dedup.rs`/`conflict.rs`/
`consolidation.rs` (depend on zeroclaw-config; revisit when the lifecycle
features land), `retrieval.rs` (an in-process cache decorator, not needed).

Author's own prior projects reused elsewhere in Ganglion (declared in the
top-level README): esclusa (HMAC ledger), coati (MCP server template).
