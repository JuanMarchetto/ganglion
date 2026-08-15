#!/usr/bin/env bash
# Bring up the local 3-node CockroachDB cluster + haproxy, init it, and
# enable the vector index feature. Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d

echo "waiting for roach1 to accept connections..."
for i in $(seq 1 60); do
  if docker compose exec -T roach1 ./cockroach sql --insecure -e "SELECT 1" >/dev/null 2>&1; then
    break
  fi
  # Not initialized yet: `cockroach sql` fails until `init` runs on a fresh cluster.
  docker compose exec -T roach1 ./cockroach init --insecure >/dev/null 2>&1 || true
  sleep 1
done

docker compose exec -T roach1 ./cockroach sql --insecure -e "SELECT 1" >/dev/null

# Vector indexing is gated behind a cluster setting in v25.x (see README).
docker compose exec -T roach1 ./cockroach sql --insecure \
  -e "SET CLUSTER SETTING feature.vector_index.enabled = true" 2>/dev/null || echo "  (feature.vector_index.enabled not needed on this version)"

docker compose exec -T roach1 ./cockroach sql --insecure \
  -e "CREATE DATABASE IF NOT EXISTS ganglion"

echo "cluster ready:"
echo "  SQL (via haproxy):  postgresql://root@localhost:26257/ganglion?sslmode=disable"
echo "  DB Console:         http://localhost:8085"
