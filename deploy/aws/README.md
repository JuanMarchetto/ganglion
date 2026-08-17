# Deploying Ganglion on AWS

One EC2 instance runs the whole stack: the Ganglion server, a 3-node
CockroachDB cluster, haproxy in front of it, and Caddy terminating TLS. Three
nodes on one host is a deliberate demo shape — it makes the kill-node
demonstration real (a node genuinely dies, and the cluster genuinely keeps
quorum) without paying for three instances. See
[Production notes](#production-notes) for what changes when the goal stops
being a demo.

> **Status: not yet executed.** Every command here is written to be run
> as-is, but nothing in this directory has been applied to a live AWS
> account — the account did not exist when this was written. Times and
> costs are estimates; everything below the AWS boundary (the compose
> stack, the failover behaviour, the harness numbers) is measured locally.

## What you need

- An AWS account with EC2 access. **Do not use a client's account.**
- A domain (or subdomain) you can point at an IP. Caddy needs it to issue a
  certificate; without it you get HTTP only.
- An SSH keypair in the target region.

## 1. Launch the instance

- **AMI:** Ubuntu Server 24.04 LTS (x86_64)
- **Type:** `t3.large` (2 vCPU / 8 GB). Three CockroachDB nodes plus a Rust
  build want the headroom; `t3.medium` works if you build the image
  elsewhere and pull it. `t3.micro` does not — three nodes will OOM.
- **Storage:** 30 GB gp3
- **Region:** `us-east-1` keeps it next to the CockroachDB Cloud cluster.

## 2. Security group

Exactly three inbound rules. The database ports are deliberately absent:
CockroachDB, its console, and the app all stay on the private docker network
and are reachable only through Caddy.

| Port | Source | Why |
|---|---|---|
| 22 | your IP only | SSH |
| 80 | 0.0.0.0/0 | HTTP, and the ACME challenge Caddy needs |
| 443 | 0.0.0.0/0 | HTTPS |

## 3. Point DNS at the instance

Create an A record for `GANGLION_DOMAIN` pointing at the instance's public
IPv4 address, and confirm it resolves before step 5 — Caddy asks Let's
Encrypt for a certificate on startup, and the ACME challenge fails if DNS
has not propagated. Allocate an Elastic IP if the host will be stopped and
started, or the address changes underneath the record.

## 4. Install Docker

```sh
ssh ubuntu@<public-ip>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit   # log back in so the group membership takes effect
```

## 5. Deploy

```sh
git clone https://github.com/JuanMarchetto/ganglion.git
cd ganglion/deploy/aws
cp env.example .env
# Fill in GANGLION_DOMAIN and GANGLION_HMAC_KEY (openssl rand -hex 32).
# For the live kill-node demo also set GANGLION_ENABLE_CHAOS=1 and
# GANGLION_ADMIN_TOKEN (openssl rand -hex 24).
nano .env

docker compose -f docker-compose.prod.yml up -d --build
```

First run takes roughly 10-15 minutes, nearly all of it compiling the Rust
release binary on a 2-vCPU box. Then:

```sh
docker compose -f docker-compose.prod.yml logs -f app
```

The app is healthy once `/healthz` answers. First boot against an empty
schema additionally spends 40-60s creating tables and building the vector
index — the DDL waits on a background job — so the container can look idle
before it starts serving. The healthcheck's `start-period` already allows
for this.

Verify from your laptop:

```sh
curl https://<your-domain>/healthz                       # -> ok
curl https://<your-domain>/api/nodes                     # -> 3 nodes, is_live true
curl -X POST https://<your-domain>/api/remember \
  -H 'Content-Type: application/json' \
  -d '{"key":"hello","content":"Ganglion is live on AWS"}'
```

Then open `https://<your-domain>/` for the UI.

## 6. The kill-node demo

With `GANGLION_ENABLE_CHAOS=1` and a token set, the UI's KILL button works
against the deployed cluster. It is gated three ways, and all three hold
regardless of what the caller sends:

1. the endpoints 404-equivalent (403) unless `GANGLION_ENABLE_CHAOS=1`;
2. a bearer token must match `GANGLION_ADMIN_TOKEN`;
3. the container name must be in a fixed allowlist compiled into the binary
   (`ganglion-roach1-1`, `-2`, `-3`), and it is passed to `docker` as an
   argument vector rather than through a shell — an arbitrary string is
   rejected by the allowlist, and could not become a command even if it
   were not.

Reviving a node is the same endpoint with `revive`:

```sh
curl -X POST https://<your-domain>/api/chaos/revive \
  -H "Authorization: Bearer $GANGLION_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' -d '{"node":"ganglion-roach2-1"}'
```

The allowlist carries both project prefixes (`local-roach*` from
`deploy/local`, `ganglion-roach*` from this stack, since compose names
containers `<project>-<service>-<index>` and this file sets
`name: ganglion`). If you rename the project, `docker ps --format
'{{.Names}}'` will disagree with `CHAOS_ALLOWLIST` in
`crates/ganglion-server/src/main.rs` and the button answers "node not in
allowlist" — harmless, but check it before a recording.

Granting the app the host's docker socket is what makes the button possible,
and it is a real trade: a process that can talk to the docker daemon can do
anything on the host. It is mounted for the demo. For any deployment where
that is not the point, set `GANGLION_ENABLE_CHAOS=0` and comment out the
socket mount in `docker-compose.prod.yml` — the API keeps working, only the
button dies.

## 7. S3 snapshots

CockroachDB backs itself up straight to S3, which is also the second AWS
service the stack uses:

```sh
aws s3 mb s3://ganglion-backups-<suffix> --region us-east-1

docker compose -f docker-compose.prod.yml exec roach1 ./cockroach sql \
  --insecure --host=roach1:26257 -e \
  "BACKUP DATABASE ganglion INTO 's3://ganglion-backups-<suffix>/ganglion?AWS_ACCESS_KEY_ID=…&AWS_SECRET_ACCESS_KEY=…'"
```

Prefer an IAM instance role over inline keys — attach a role with
`s3:PutObject` on that bucket and drop the credentials from the URL. Keys in
a shell command land in the host's shell history and in the cluster's job
table.

## Operations

```sh
# state of the world
docker compose -f docker-compose.prod.yml ps

# app logs
docker compose -f docker-compose.prod.yml logs -f app

# SQL shell
docker compose -f docker-compose.prod.yml exec roach1 \
  ./cockroach sql --insecure --host=roach1:26257 --database=ganglion

# redeploy after a git pull
git pull && docker compose -f docker-compose.prod.yml up -d --build app

# full stop (data survives in named volumes)
docker compose -f docker-compose.prod.yml down
```

## Cost

A `t3.large` runs about $60/month on-demand in us-east-1, plus ~$2.40 for
30 GB gp3 and a few cents of S3. Stop the instance between demos and you pay
only for storage. Set a billing alarm before leaving it running — this is
the account's first workload, so there is no baseline to notice a change
against.

## Production notes

What this shape is honest about, and what would change:

- **Three nodes, one host.** Survives a process death, which is the point of
  the demo, but not an instance or AZ failure. Real deployment: one node per
  AZ across three, or CockroachDB Cloud with a multi-region cluster.
- **`--insecure`.** No TLS between nodes and no SQL authentication; the
  cluster is unreachable from outside the docker network, which is what makes
  this tolerable rather than safe. Production wants certificates and a real
  SQL user.
- **haproxy is a single point of failure.** For the kill-node story it is
  fine, since the thing being killed is a database node. Removing it means a
  client that knows all three node addresses, or an NLB.
- **`GANGLION_HMAC_KEY` sits in `.env` on the host.** Anyone who can read that
  file can forge ledger entries. Secrets Manager or SSM Parameter Store is
  the fix.
