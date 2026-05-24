//! Token sampling strategies for text generation.
//!
//! - **Greedy**: always pick the highest-probability token (deterministic).
//! - **Top-K**: sample from the top K tokens.
//! - **Top-P (nucleus)**: sample from the smallest set whose cumulative
//!   probability exceeds P.
//! - **Temperature**: scale logits before softmax.

use sapient_core::error::{Result, SapientError};

// ── SamplingStrategy ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Always pick the argmax token — fastest, deterministic.
    Greedy,
    /// Sample with temperature scaling.
    Temperature(f32),
    /// Sample from the top-k highest probability tokens.
    TopK { k: usize, temperature: f32 },
    /// Nucleus sampling — sample from the minimum set covering probability p.
    TopP { p: f32, temperature: f32 },
    /// Combined top-k + top-p + temperature.
    Combined { top_k: usize, top_p: f32, temperature: f32, repetition_penalty: f32 },
}

impl Default for SamplingStrategy {
    fn default() -> Self { Self::Greedy }
}

// ── Sampler ───────────────────────────────────────────────────────────────────

pub struct Sampler {
    pub strategy: SamplingStrategy,
    rng_seed: u64,
    counter: u64,
}

impl Sampler {
    pub fn new(strategy: SamplingStrategy) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self { strategy, rng_seed: seed, counter: 0 }
    }

    pub fn with_seed(strategy: SamplingStrategy, seed: u64) -> Self {
        Self { strategy, rng_seed: seed, counter: 0 }
    }

    /// Sample the next token from `logits` — shape: (vocab_size,).
    /// Optionally pass previously generated `token_ids` for repetition penalty.
    pub fn sample(&mut self, logits: &[f32], prev_tokens: &[u32]) -> Result<u32> {
        match &self.strategy {
            SamplingStrategy::Greedy => Ok(argmax(logits)),

            SamplingStrategy::Temperature(t) => {
                let t = *t;
                let scaled = scale_logits(logits, t);
                let probs = softmax(&scaled);
                Ok(self.random_sample(&probs))
            }

            SamplingStrategy::TopK { k, temperature } => {
                let (k, t) = (*k, *temperature);
                let scaled = scale_logits(logits, t);
                let filtered = top_k_filter(&scaled, k);
                let probs = softmax(&filtered);
                Ok(self.random_sample(&probs))
            }

            SamplingStrategy::TopP { p, temperature } => {
                let (p, t) = (*p, *temperature);
                let scaled = scale_logits(logits, t);
                let filtered = top_p_filter(&scaled, p);
                let probs = softmax(&filtered);
                Ok(self.random_sample(&probs))
            }

            SamplingStrategy::Combined { top_k, top_p, temperature, repetition_penalty } => {
                let (k, p, t, rp) = (*top_k, *top_p, *temperature, *repetition_penalty);
                let mut penalized = apply_repetition_penalty(logits, prev_tokens, rp);
                penalized = scale_logits(&penalized, t);
                penalized = top_k_filter(&penalized, k);
                penalized = top_p_filter(&penalized, p);
                let probs = softmax(&penalized);
                Ok(self.random_sample(&probs))
            }
        }
    }

    /// Simple xorshift RNG for sampling without an external rand crate.
    fn random_u64(&mut self) -> u64 {
        self.counter += 1;
        let mut x = self.rng_seed.wrapping_add(self.counter.wrapping_mul(6364136223846793005));
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        x
    }

    fn random_f32(&mut self) -> f32 {
        (self.random_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    fn random_sample(&mut self, probs: &[f32]) -> u32 {
        let r = self.random_f32();
        let mut cum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    out.iter_mut().for_each(|x| *x /= sum);
    out
}

fn scale_logits(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature <= 0.0 || temperature == 1.0 {
        return logits.to_vec();
    }
    logits.iter().map(|&x| x / temperature).collect()
}

fn top_k_filter(logits: &[f32], k: usize) -> Vec<f32> {
    if k == 0 || k >= logits.len() { return logits.to_vec(); }
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = indexed[k - 1].1;
    logits.iter().map(|&x| if x >= threshold { x } else { f32::NEG_INFINITY }).collect()
}

fn top_p_filter(logits: &[f32], p: f32) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let probs = softmax(logits);
    let mut sorted_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cum = 0.0f32;
    let mut cutoff_idx = sorted_probs.len();
    for (i, (_, prob)) in sorted_probs.iter().enumerate() {
        cum += prob;
        if cum >= p {
            cutoff_idx = i + 1;
            break;
        }
    }

    let keep: std::collections::HashSet<usize> = sorted_probs[..cutoff_idx].iter().map(|(i, _)| *i).collect();
    logits.iter().enumerate().map(|(i, &x)| if keep.contains(&i) { x } else { f32::NEG_INFINITY }).collect()
}

fn apply_repetition_penalty(logits: &[f32], prev_tokens: &[u32], penalty: f32) -> Vec<f32> {
    if (penalty - 1.0).abs() < 1e-6 { return logits.to_vec(); }
    let mut out = logits.to_vec();
    for &tok in prev_tokens {
        let idx = tok as usize;
        if idx < out.len() {
            if out[idx] >= 0.0 {
                out[idx] /= penalty;
            } else {
                out[idx] *= penalty;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1, 0.9, 0.3, 0.5];
        let mut s = Sampler::with_seed(SamplingStrategy::Greedy, 42);
        assert_eq!(s.sample(&logits, &[]).unwrap(), 1);
    }

    #[test]
    fn top_k_removes_low_prob() {
        let logits = vec![10.0, 1.0, 1.0, 1.0];
        let filtered = top_k_filter(&logits, 1);
        assert_eq!(filtered[0], 10.0);
        assert!(filtered[1].is_infinite() && filtered[1] < 0.0);
    }

    #[test]
    fn repetition_penalty_reduces_score() {
        let logits = vec![1.0, 2.0, 3.0];
        let penalized = apply_repetition_penalty(&logits, &[2], 1.3);
        assert!(penalized[2] < logits[2]);
    }
}
