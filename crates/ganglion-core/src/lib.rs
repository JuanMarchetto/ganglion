//! Ganglion core — distributed agent memory on CockroachDB.
//!
//! Portions vendored from zeroclaw (MIT OR Apache-2.0); see ATTRIBUTION.md.

pub mod chunker;
pub mod cockroach;
pub mod embeddings;
pub mod ledger;
pub mod traits;
pub mod vector;

pub use cockroach::{
    BeliefVersion, BeliefWritten, CockroachMemory, Edge, LedgerVerification, RowMismatch,
};
pub use ledger::{ChainReport, LedgerEntry, verify_chain};
pub use embeddings::{
    EmbeddingProvider, HashEmbedding, NoopEmbedding, OpenAiEmbedding, create_embedding_provider,
};
pub use traits::{
    ExportFilter, Memory, MemoryCategory, MemoryEntry, MemoryKind, MemoryStats, SemanticSubtype,
    StoreOptions, normalize_recent_recall_query, sanitize_session_key,
};
