//! Tamper-evident memory ledger: canonical string + HMAC-SHA256 chain.
//!
//! Pure chain logic (canonical form, HMAC, verification) adapted from
//! esclusa's `gate/src/ledger.rs` (author's own prior code — see
//! ATTRIBUTION.md), re-fielded for the memory domain and stripped of the
//! Postgres advisory-lock plumbing: CockroachDB does not implement
//! `pg_advisory_xact_lock` (verified empirically on v26.2.5), so append
//! serialization instead relies on SERIALIZABLE isolation — two concurrent
//! appends both read the same head, one commits, the other surfaces 40001
//! and is retried by `with_txn_retry`.
//!
//! Every memory mutation (store / supersede / forget / purge / rename)
//! appends one entry *in the same transaction* as the mutation itself,
//! committing to the mutated row via [`row_sha256`] in `params`. Entries
//! chain **per (tenant, agent) scope** ([`chain_for`]; bulk ops use
//! [`SYSTEM_CHAIN`]) — a global chain would funnel all writers through one
//! hot head row. Tampering with a ledger row breaks that scope's HMAC chain
//! at its id; tampering with a memory row (content, but also `valid_to`,
//! `key`, `importance`, `pinned`) makes the row's hash disagree with its
//! latest ledger entry. Both are caught by
//! [`crate::cockroach::CockroachMemory::verify_ledger`].

use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Chain anchor for the first entry.
pub const GENESIS: &str = "genesis";

/// Chain for mutations with no tenant/agent scope (bulk purges, renames).
pub const SYSTEM_CHAIN: &str = "system";

/// Ledger chain id for a (tenant, agent) scope. Ledger entries chain **per
/// scope**, not globally: a global chain funnels every writer through one
/// hot head row and, under sustained concurrency, starves writers into
/// user-visible errors after retry exhaustion — the exact failure the ledger
/// exists to rule out. Per-scope chains keep the tamper-evidence property
/// (each scope's history is a complete HMAC chain) without the hot key.
/// Length-prefixed for injectivity, same argument as [`canonical_string`].
pub fn chain_for(tenant_id: &str, agent_id: &str) -> String {
    let mut out = String::with_capacity(tenant_id.len() + agent_id.len() + 8);
    push_field(&mut out, tenant_id);
    push_field(&mut out, agent_id);
    out
}

/// Environment variable holding the ledger signing key.
pub const HMAC_KEY_ENV: &str = "GANGLION_HMAC_KEY";

/// Development-only fallback signing key.
pub const DEV_KEY: &str = "ganglion-dev-key-not-for-production";

/// Read the signing key from `GANGLION_HMAC_KEY`, falling back to the dev key
/// with a loud warning — a known key makes the chain forgeable.
pub fn key_from_env() -> Vec<u8> {
    match std::env::var(HMAC_KEY_ENV) {
        Ok(k) if !k.is_empty() => k.into_bytes(),
        _ => {
            tracing::warn!(
                "{HMAC_KEY_ENV} not set — using the built-in dev key; \
                 the ledger is tamper-EVIDENT only against parties without the key"
            );
            DEV_KEY.as_bytes().to_vec()
        }
    }
}

/// One ledger row. `ts` is the RFC 3339 microsecond string that entered the
/// canonical form (timestamptz stores microseconds, so it round-trips).
#[derive(Clone, Debug, serde::Serialize)]
pub struct LedgerEntry {
    /// Scope chain this entry belongs to ([`chain_for`] / [`SYSTEM_CHAIN`]).
    /// `id` and `prev_hmac` are per-chain.
    pub chain_id: String,
    pub id: i64,
    pub ts: String,
    /// Mutation kind: `store` | `supersede` | `forget` | `purge`.
    pub kind: String,
    /// Agent alias (or "system") that performed the mutation.
    pub actor: String,
    /// Memory row id the mutation targeted ("*" for bulk purges).
    pub target: String,
    /// Mutation details; includes `content_sha256` for content-bearing kinds.
    pub params: Value,
    pub prev_hmac: String,
    pub hmac: String,
}

/// Append one field as `<byte-length>:<bytes>|`. The length prefix makes the
/// concatenation injective: without it a `|` planted inside one free-form
/// field could shift bytes into the next field and produce a different row
/// that still recomputes to the stored HMAC.
fn push_field(out: &mut String, s: &str) {
    out.push_str(&s.len().to_string());
    out.push(':');
    out.push_str(s);
    out.push('|');
}

/// Canonical byte string covered by the HMAC. `chain_id` is part of the
/// signed material so an entry cannot be spliced into a different chain.
#[allow(clippy::too_many_arguments)]
pub fn canonical_string(
    chain_id: &str,
    id: i64,
    ts: &str,
    kind: &str,
    actor: &str,
    target: &str,
    params: &Value,
    prev_hmac: &str,
) -> String {
    // serde_json object maps are ordered, so compact serialization is
    // deterministic across the insert and verify paths.
    let params_json = serde_json::to_string(params).unwrap_or_else(|_| "null".to_string());
    let mut out = String::with_capacity(224);
    push_field(&mut out, chain_id);
    push_field(&mut out, &id.to_string());
    push_field(&mut out, ts);
    push_field(&mut out, kind);
    push_field(&mut out, actor);
    push_field(&mut out, target);
    push_field(&mut out, &params_json);
    push_field(&mut out, prev_hmac);
    out
}

pub fn entry_canonical(e: &LedgerEntry) -> String {
    canonical_string(
        &e.chain_id,
        e.id,
        &e.ts,
        &e.kind,
        &e.actor,
        &e.target,
        &e.params,
        &e.prev_hmac,
    )
}

/// JSONB normalizes float negative zero on the way in; canonicalize before
/// signing so insert-time and verify-time strings agree.
pub fn normalize_params(v: &Value) -> Value {
    match v {
        Value::Number(n) if n.is_f64() => match n.as_f64() {
            Some(f) if f == 0.0 && f.is_sign_negative() => serde_json::Number::from_f64(0.0)
                .map(Value::Number)
                .unwrap_or_else(|| v.clone()),
            _ => v.clone(),
        },
        Value::Array(a) => Value::Array(a.iter().map(normalize_params).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, x)| (k.clone(), normalize_params(x)))
                .collect(),
        ),
        _ => v.clone(),
    }
}

pub fn hmac_hex(key: &[u8], msg: &str) -> String {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts keys of any length");
    mac.update(msg.as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Hex SHA-256 of a memory row's content — the commitment stored in `params`.
pub fn content_sha256(content: &str) -> String {
    hex::encode(Sha256::digest(content.as_bytes()))
}

/// Hex SHA-256 committing to the full tamper-relevant state of a memory row,
/// not just its `content`: an attacker who flips `valid_to` (resurrecting a
/// superseded belief), edits `key`, or toggles `pinned`/`importance` changes
/// this hash. Timestamps must be RFC 3339 **microsecond** strings
/// (`ts_micros_string`) so the hash survives a database round-trip; `None`
/// hashes as the empty field. Length-prefixed fields, same injectivity
/// argument as [`canonical_string`].
pub fn row_sha256(
    key: &str,
    content: &str,
    valid_from: &str,
    valid_to: Option<&str>,
    importance: Option<f64>,
    pinned: bool,
) -> String {
    let mut out = String::with_capacity(160 + content.len());
    push_field(&mut out, key);
    push_field(&mut out, content);
    push_field(&mut out, valid_from);
    push_field(&mut out, valid_to.unwrap_or(""));
    let importance_s = importance.map(|f| f.to_string()).unwrap_or_default();
    push_field(&mut out, &importance_s);
    push_field(&mut out, if pinned { "true" } else { "false" });
    hex::encode(Sha256::digest(out.as_bytes()))
}

/// Chain verification result.
#[derive(Debug, serde::Serialize)]
pub struct ChainReport {
    pub valid: bool,
    pub checked: usize,
    pub head_hmac: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub broken_at: Option<i64>,
}

/// Recompute the whole chain. `entries` must be ordered by id ascending.
pub fn verify_chain(key: &[u8], entries: &[LedgerEntry]) -> ChainReport {
    let head_hmac = entries.last().map(|e| e.hmac.clone());
    let mut prev = GENESIS.to_string();
    for e in entries {
        let linked = e.prev_hmac == prev;
        let recomputed = hmac_hex(key, &entry_canonical(e));
        if !linked || recomputed != e.hmac {
            return ChainReport {
                valid: false,
                checked: entries.len(),
                head_hmac,
                broken_at: Some(e.id),
            };
        }
        prev = e.hmac.clone();
    }
    ChainReport {
        valid: true,
        checked: entries.len(),
        head_hmac,
        broken_at: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const KEY: &[u8] = b"test-signing-key";

    fn build(id: i64, prev: &str, target: &str, params: Value) -> LedgerEntry {
        let mut e = LedgerEntry {
            chain_id: chain_for("t1", "agent-1"),
            id,
            ts: format!("2026-08-15T12:00:0{id}.000000Z"),
            kind: "store".into(),
            actor: "tester".into(),
            target: target.into(),
            params,
            prev_hmac: prev.into(),
            hmac: String::new(),
        };
        e.hmac = hmac_hex(KEY, &entry_canonical(&e));
        e
    }

    fn chain_of_three() -> Vec<LedgerEntry> {
        let e1 = build(1, GENESIS, "row-a", json!({"key": "k1"}));
        let e2 = build(2, &e1.hmac, "row-b", json!({"key": "k2"}));
        let e3 = build(3, &e2.hmac, "row-a", json!({"key": "k1", "supersedes": "row-a"}));
        vec![e1, e2, e3]
    }

    #[test]
    fn canonical_string_format_is_pinned() {
        let c = canonical_string(
            "chain-a",
            7,
            "2026-08-15T12:00:00.000000Z",
            "store",
            "agent",
            "row-1",
            &json!({"x": 1}),
            GENESIS,
        );
        assert_eq!(
            c,
            "7:chain-a|1:7|27:2026-08-15T12:00:00.000000Z|5:store|5:agent|5:row-1|7:{\"x\":1}|7:genesis|"
        );
    }

    #[test]
    fn same_entry_in_a_different_chain_has_a_different_hmac() {
        let chain = chain_of_three();
        let mut moved = chain[0].clone();
        moved.chain_id = chain_for("t2", "agent-1");
        assert_ne!(
            hmac_hex(KEY, &entry_canonical(&chain[0])),
            hmac_hex(KEY, &entry_canonical(&moved))
        );
    }

    #[test]
    fn chain_for_is_injective_across_field_boundaries() {
        assert_ne!(chain_for("ab", "c"), chain_for("a", "bc"));
    }

    #[test]
    fn row_sha256_covers_validity_and_flags() {
        let base = row_sha256("k", "c", "2026-08-15T12:00:00.000000Z", None, None, false);
        assert_ne!(
            base,
            row_sha256(
                "k",
                "c",
                "2026-08-15T12:00:00.000000Z",
                Some("2026-08-15T13:00:00.000000Z"),
                None,
                false
            ),
            "valid_to must change the hash"
        );
        assert_ne!(
            base,
            row_sha256("k", "c", "2026-08-15T12:00:00.000000Z", None, Some(0.5), false),
            "importance must change the hash"
        );
        assert_ne!(
            base,
            row_sha256("k", "c", "2026-08-15T12:00:00.000000Z", None, None, true),
            "pinned must change the hash"
        );
    }

    /// Planting `|` in a free-form field must not let a row be rewritten into
    /// a different meaning that recomputes to the same HMAC.
    #[test]
    fn delimiter_in_a_field_cannot_forge_a_matching_canonical_string() {
        let planted = canonical_string(
            "chain-a",
            5,
            "2026-08-15T12:00:00.000000Z",
            "store",
            "mallory|forget|row-9",
            "row-1",
            &json!({}),
            "prev",
        );
        let rewritten = canonical_string(
            "chain-a",
            5,
            "2026-08-15T12:00:00.000000Z",
            "store",
            "mallory",
            "forget|row-9|row-1",
            &json!({}),
            "prev",
        );
        assert_ne!(planted, rewritten);
        assert_ne!(hmac_hex(KEY, &planted), hmac_hex(KEY, &rewritten));
    }

    #[test]
    fn chain_of_three_verifies() {
        let chain = chain_of_three();
        let report = verify_chain(KEY, &chain);
        assert!(report.valid);
        assert_eq!(report.checked, 3);
        assert!(report.broken_at.is_none());
    }

    #[test]
    fn empty_chain_is_valid() {
        let report = verify_chain(KEY, &[]);
        assert!(report.valid);
        assert_eq!(report.checked, 0);
    }

    #[test]
    fn tampered_field_breaks_at_that_id() {
        let mut chain = chain_of_three();
        chain[1].target = "row-x".into();
        let report = verify_chain(KEY, &chain);
        assert!(!report.valid);
        assert_eq!(report.broken_at, Some(2));
    }

    #[test]
    fn rewritten_history_cannot_relink_without_the_key() {
        let mut chain = chain_of_three();
        chain[1].params = json!({"key": "forged"});
        chain[2].prev_hmac = chain[1].hmac.clone();
        let report = verify_chain(KEY, &chain);
        assert!(!report.valid);
        assert_eq!(report.broken_at, Some(2));
    }

    #[test]
    fn wrong_key_breaks_at_genesis() {
        let chain = chain_of_three();
        let report = verify_chain(b"other-key", &chain);
        assert!(!report.valid);
        assert_eq!(report.broken_at, Some(1));
    }

    #[test]
    fn negative_zero_is_normalized_before_signing() {
        let raw: Value = serde_json::from_str("{\"n\": -0.0}").unwrap();
        assert_eq!(
            serde_json::to_string(&normalize_params(&raw)).unwrap(),
            "{\"n\":0.0}"
        );
        let ints: Value = serde_json::from_str("{\"c\": 0}").unwrap();
        assert_eq!(
            serde_json::to_string(&normalize_params(&ints)).unwrap(),
            "{\"c\":0}"
        );
    }

    #[test]
    fn content_sha256_is_stable_hex() {
        assert_eq!(
            content_sha256("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }
}
