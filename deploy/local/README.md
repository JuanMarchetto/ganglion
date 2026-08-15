# Local 3-node CockroachDB cluster

```
./up.sh        # idempotent: up + init + settings + ganglion database
docker compose down       # stop (keep data)
docker compose down -v    # destroy including data
```

- SQL via haproxy: `postgresql://root@localhost:26257/ganglion?sslmode=disable`
- DB Console: http://localhost:8085
- Kill-node drill: `docker compose kill roach2` — haproxy health checks
  (`/health?ready=1`) stop routing to it within seconds; `docker compose start roach2` heals.

## Empirical compatibility notes (verified on `latest-v26.2` = v26.2.5, 2026-08-15)

- `VECTOR(n)` is native — **no** `CREATE EXTENSION vector`.
- `CREATE VECTOR INDEX ... (tenant_id, agent_id, embedding)` with prefix columns
  works. On v26.2 `feature.vector_index.enabled` defaults to **true** (also
  confirmed on CockroachDB Cloud serverless v26.2.5); on v25.x it must be set,
  so `up.sh` still applies it, tolerating failure.
- Cosine distance operator `<=>` works and the vector index serves
  `ORDER BY embedding <=> $q LIMIT k` (verified on v25.4 and v26.2).
- FTS: `to_tsvector`, `plainto_tsquery`, `@@` and `ts_rank` work; **`ts_rank_cd`
  is unimplemented** (SQLSTATE 0A000) — the port uses `ts_rank` instead.
- GIN expression indexes (`USING gin (to_tsvector('simple', content))`) work.
- `x = ANY(array)` works.
