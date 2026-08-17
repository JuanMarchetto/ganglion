# Two stages: build the release binary, then ship it on a slim runtime.
# The UI is compiled into the binary (include_str!), so the runtime image
# carries no static assets and there is nothing to serve from disk.

FROM rust:1-bookworm AS build
WORKDIR /src

# Copy manifests first so dependency compilation caches across code edits.
COPY Cargo.toml Cargo.lock ./
COPY crates/ganglion-core/Cargo.toml crates/ganglion-core/
COPY crates/ganglion-mcp/Cargo.toml crates/ganglion-mcp/
COPY crates/ganglion-server/Cargo.toml crates/ganglion-server/

COPY crates crates
RUN cargo build --release -p ganglion-server --bin ganglion-server

FROM debian:bookworm-slim
# ca-certificates for TLS to CockroachDB Cloud; curl for the healthcheck;
# docker-cli only so the gated kill-node endpoint can reach the mounted
# docker socket. Drop docker.io from this list if you deploy with
# GANGLION_ENABLE_CHAOS=0 and no socket mount.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl docker.io \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/target/release/ganglion-server /usr/local/bin/ganglion-server

EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=6 \
  CMD curl -fsS http://localhost:3000/healthz || exit 1

# start-period is generous on purpose: first boot against an empty schema
# creates the tables and the vector index, and the index build is a
# background job the DDL waits on — measured at 40-60s locally.

CMD ["ganglion-server"]
