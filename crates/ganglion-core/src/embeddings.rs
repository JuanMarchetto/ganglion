//! Embedding providers.
//!
//! Vendored from zeroclaw `zeroclaw-memory/src/embeddings.rs` (MIT OR
//! Apache-2.0), with zeroclaw's logging/config plumbing replaced by
//! `tracing` and a plain reqwest client. See ATTRIBUTION.md.

use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingIdentity {
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
}

impl std::fmt::Display for EmbeddingIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}/{} ({} dims)",
            self.provider, self.model, self.dimensions
        )
    }
}

/// Trait for embedding providers — convert text to vectors
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Provider name
    fn name(&self) -> &str;

    /// Embedding dimensions
    fn dimensions(&self) -> usize;

    /// Embed a batch of texts into vectors
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed a single text
    async fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut results = self.embed(&[text]).await?;
        results.pop().ok_or_else(|| {
            tracing::error!("embed_one: provider returned no embedding");
            anyhow::Error::msg("Empty embedding result")
        })
    }
}

// ── Noop provider (keyword-only fallback) ────────────────────

pub struct NoopEmbedding;

#[async_trait]
impl EmbeddingProvider for NoopEmbedding {
    fn name(&self) -> &str {
        "none"
    }

    fn dimensions(&self) -> usize {
        0
    }

    async fn embed(&self, _texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }
}

// ── OpenAI-compatible embedding provider ─────────────────────

pub struct OpenAiEmbedding {
    base_url: String,
    api_key: String,
    model: String,
    dims: usize,
}

impl OpenAiEmbedding {
    pub fn new(base_url: &str, api_key: &str, model: &str, dims: usize) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            dims,
        }
    }

    fn http_client(&self) -> reqwest::Client {
        reqwest::Client::new()
    }

    fn has_explicit_api_path(&self) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.base_url) else {
            return false;
        };

        let path = url.path().trim_end_matches('/');
        !path.is_empty() && path != "/"
    }

    fn has_embeddings_endpoint(&self) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.base_url) else {
            return false;
        };

        url.path().trim_end_matches('/').ends_with("/embeddings")
    }

    fn embeddings_url(&self) -> String {
        if self.has_embeddings_endpoint() {
            return self.base_url.clone();
        }

        if self.has_explicit_api_path() {
            format!("{}/embeddings", self.base_url)
        } else {
            format!("{}/v1/embeddings", self.base_url)
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbedding {
    fn name(&self) -> &str {
        "openai"
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });

        let resp = self
            .http_client()
            .post(self.embeddings_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Embedding API error {status}: {text}");
        }

        let json: serde_json::Value = resp.json().await?;
        let data = json.get("data").and_then(|d| d.as_array()).ok_or_else(|| {
            tracing::error!("embedding response missing 'data' field");
            anyhow::Error::msg("Invalid embedding response: missing 'data'")
        })?;

        let mut embeddings = Vec::with_capacity(data.len());
        for item in data {
            let embedding = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    tracing::error!("embedding response item missing 'embedding' array");
                    anyhow::Error::msg("Invalid embedding item")
                })?;

            #[allow(clippy::cast_possible_truncation)]
            let vec: Vec<f32> = embedding
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            embeddings.push(vec);
        }

        Ok(embeddings)
    }
}

// ── Deterministic local provider (no network) ────────────────
//
// New in Ganglion: a hashed bag-of-words embedding. Deterministic and
// dependency-free, so vector recall is exercisable in tests and demos
// without an external embedding service. Not a semantic model — texts
// sharing tokens land near each other, nothing more.

pub struct HashEmbedding {
    dims: usize,
}

impl HashEmbedding {
    pub fn new(dims: usize) -> Self {
        Self { dims: dims.max(1) }
    }
}

#[async_trait]
impl EmbeddingProvider for HashEmbedding {
    fn name(&self) -> &str {
        "hash"
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0_f32; self.dims];
                for token in text
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|t| !t.is_empty())
                {
                    let mut h = DefaultHasher::new();
                    token.hash(&mut h);
                    #[allow(clippy::cast_possible_truncation)]
                    let bucket = (h.finish() % self.dims as u64) as usize;
                    v[bucket] += 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > f32::EPSILON {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect())
    }
}

// ── Factory ──────────────────────────────────────────────────

pub fn create_embedding_provider(
    model_provider: &str,
    api_key: Option<&str>,
    model: &str,
    dims: usize,
) -> Box<dyn EmbeddingProvider> {
    match model_provider {
        "openai" => {
            let key = api_key.unwrap_or("");
            Box::new(OpenAiEmbedding::new(
                "https://api.openai.com",
                key,
                model,
                dims,
            ))
        }
        "openrouter" => {
            let key = api_key.unwrap_or("");
            Box::new(OpenAiEmbedding::new(
                "https://openrouter.ai/api/v1",
                key,
                model,
                dims,
            ))
        }
        "hash" => Box::new(HashEmbedding::new(dims)),
        name if name.starts_with("custom:") => {
            let base_url = name.strip_prefix("custom:").unwrap_or("");
            let key = api_key.unwrap_or("");
            Box::new(OpenAiEmbedding::new(base_url, key, model, dims))
        }
        _ => Box::new(NoopEmbedding),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_name() {
        let p = NoopEmbedding;
        assert_eq!(p.name(), "none");
        assert_eq!(p.dimensions(), 0);
    }

    #[tokio::test]
    async fn noop_embed_returns_empty() {
        let p = NoopEmbedding;
        let result = p.embed(&["hello"]).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn factory_none() {
        let p = create_embedding_provider("none", None, "model", 1536);
        assert_eq!(p.name(), "none");
    }

    #[test]
    fn factory_openai() {
        let p = create_embedding_provider("openai", Some("key"), "text-embedding-3-small", 1536);
        assert_eq!(p.name(), "openai");
        assert_eq!(p.dimensions(), 1536);
    }

    #[test]
    fn factory_hash() {
        let p = create_embedding_provider("hash", None, "", 64);
        assert_eq!(p.name(), "hash");
        assert_eq!(p.dimensions(), 64);
    }

    #[tokio::test]
    async fn hash_embedding_is_deterministic_and_normalized() {
        let p = HashEmbedding::new(32);
        let a = p.embed_one("hola mundo distribuido").await.unwrap();
        let b = p.embed_one("hola mundo distribuido").await.unwrap();
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn hash_embedding_shared_tokens_are_closer() {
        let p = HashEmbedding::new(64);
        let base = p.embed_one("resilient distributed memory").await.unwrap();
        let near = p.embed_one("distributed memory survives").await.unwrap();
        let far = p.embed_one("banana pancake recipe").await.unwrap();
        let sim_near = crate::vector::cosine_similarity(&base, &near);
        let sim_far = crate::vector::cosine_similarity(&base, &far);
        assert!(
            sim_near > sim_far,
            "shared-token text should be closer: near={sim_near} far={sim_far}"
        );
    }

    #[test]
    fn embeddings_url_standard_openai() {
        let p = OpenAiEmbedding::new("https://api.openai.com", "key", "model", 1536);
        assert_eq!(p.embeddings_url(), "https://api.openai.com/v1/embeddings");
    }

    #[test]
    fn embeddings_url_base_with_v1_no_duplicate() {
        let p = OpenAiEmbedding::new("https://api.example.com/v1", "key", "model", 1536);
        assert_eq!(p.embeddings_url(), "https://api.example.com/v1/embeddings");
    }

    #[test]
    fn embeddings_url_custom_full_endpoint() {
        let p = OpenAiEmbedding::new(
            "https://my-api.example.com/api/v2/embeddings",
            "key",
            "model",
            1536,
        );
        assert_eq!(
            p.embeddings_url(),
            "https://my-api.example.com/api/v2/embeddings"
        );
    }
}
