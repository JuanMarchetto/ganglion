#!/usr/bin/env bash
# Stop everything this project runs on the machine: the cluster, and any
# server/harness process left behind by a demo or an interrupted run.
# Data survives in the named volumes, so up.sh brings the same cluster back.
set -uo pipefail
cd "$(dirname "$0")"

pkill -f 'target/(debug|release)/(ganglion-server|harness)' 2>/dev/null && echo "stopped ganglion processes"

docker compose down

echo "load average now: $(cut -d' ' -f1-3 /proc/loadavg)"
