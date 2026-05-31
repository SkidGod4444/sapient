# `sapient serve` — serving architecture & roadmap

Goal: a local OpenAI-compatible server that beats Ollama — fast model switching,
high concurrency, low latency — across CPU-only, GPU-only, and hybrid hardware.
Grounded in the deep-research report (vLLM, llama.cpp, mistral.rs, TensorRT-LLM).

**Measured results:** see [SERVING_BENCHMARKS.md](SERVING_BENCHMARKS.md) — on an
Apple M4 (Metal), `sapient serve` beats Ollama on TTFT (4.2×), decode (1.25×),
concurrent throughput (1.31×), and model switch-back (6×). vLLM is a datacenter-GPU
engine and doesn't run on this edge box. Harness: `scripts/bench_compete.py`.

## Built (v0.3.x)

### Phase 1 — Multi-model LRU residency
`ModelCache` in `server.rs`. Ollama keeps **one** model resident and cold-reloads
on every switch; we keep the **N most-recently-used** models resident, bounded by
a RAM budget.
- `--max-models N` (default 3) + `--cache-gb X` (default ~70% of system RAM). Evict
  LRU until **both** the count and byte budgets fit; the just-used model is never
  evicted. Size estimated from on-disk weight size (`hub::cached_model_size`).
- Each entry is `Arc<CachedModel>` — a streaming request keeps its model alive even
  if evicted mid-stream.
- `get_or_load` never holds the cache lock during the (slow) load or during
  inference → cache hits and *other* models' requests run concurrently. A
  `load_lock` prevents double-loading the same model on concurrent first-requests.
- `/v1/health` + `/v1/models` report `resident_models` + `active_model` (MRU).
- Measured: ~5× faster switch-back on a cache hit (no download / re-quant / engine
  rebuild). With mmap'd weights, even a cache miss that's still in the OS page cache
  reloads fast.

### Phase 3 — Admission control
`inference_sem` (tokio `Semaphore`, `--max-concurrency`, default = CPU count capped
at 8). Bounds concurrent inferences so a burst queues fairly instead of
oversubscribing the CPU/GPU. The permit is held for a request's whole lifetime
(moved into the streaming task). Note: inference is already off the async runtime
(`spawn_blocking` for streaming) and per-model-serialized (the engine lock is held
for an entire generation).

### Phase 4 — Prefix / prompt KV caching
Every generation used to `reset_cache()` + re-prefill the **entire** prompt. Now,
when `Pipeline::enable_prefix_cache()` is set (serve enables it), generation reuses
the KV cache for the longest **token** prefix shared with the previous call and only
prefills the new suffix. Multi-turn chat and shared system prompts skip re-prefilling
the whole history.
- Engine primitive: `ForwardEngine::truncate_cache(n) -> usize` (keep first `n` KV
  positions; Llama/Phi supported; MLX falls back to reset → no reuse, still correct).
- `Pipeline` tracks `last_prompt` (tokens currently in the KV); `common_prefix_len`
  finds the reuse point; only forwards `prompt[P..]`.
- **Correctness:** reuse only matching token IDs → KV at `[0..P]` is identical to a
  fresh prefill, so greedy output is byte-identical (verified: same prompt yields
  identical text cold vs. reused). Safe because same-model calls are serialized on
  the engine lock; a non-matching prefix simply falls back to a full prefill.
- Off by default (CLI chat is byte-identical to before); only `serve` enables it.

### Phase 2 — Speculative decoding in serve
`sapient serve --speculative [--draft-model <alias>]` serves every model with
speculative decoding. `SpeculativePipeline` was refactored to be serve-ready:

- **Engine reuse.** It used to rebuild *both* target+draft engines from scratch on
  every request (no engine reuse), so TTFT paid the full load+re-quantize cost per
  request. Now it holds the target and draft as `Pipeline`s and **reuses** their
  loaded `Arc<Mutex<ForwardEngine>>` inside `spawn_blocking` (locks instead of
  reloads), exactly like `Pipeline`. One-time load cost; subsequent requests are
  instant. (`Pipeline::engine_arc()` exposes the engine `Arc` for this.)
- **Per-request config + accessors.** Added `generate_with_config`,
  `chat_with_config`, `generate_stream_with_config`, `chat_stream_with_config`
  (honor `max_tokens`/`temperature`/`stop`) and `tokenizer()`/`arch()`/`is_mmap()`/
  `config()`/`format_chat_prompt()`, plus `new_with_opts`/`with_auto_draft_with_opts`
  (so serve passes `--backend`/`--mmap`).
- **Cache integration.** A resident model is now `ServedModel::{Plain(Pipeline),
  Speculative(SpeculativePipeline)}`; the LRU cache, admission control, and route
  handlers treat both uniformly. Speculative residency bills target + draft bytes.
- **Correctness fix (was fundamentally broken).** The target verification used
  `forward_all_logits` (`use_cache=false`), which **reset the KV cache and ran the
  draft tokens with no prompt context** — every speculative reply was token-salad.
  The verification is now cache-aware: `forward_all_logits_cached` appends the draft
  tokens to the target KV (positions continue from the prompt), and rejected
  speculative tokens are rolled back with `truncate_cache(n)`. The loop maintains the
  invariant "both caches hold exactly the committed tokens" and tracks each model's
  next-token logits across rounds. Verified: coherent output (e.g. *"The capital of
  France is Paris."*) on a Qwen2.5-1.5B target + Qwen2.5-0.5B draft.
- **Vocab guard + family-aware auto-draft.** Speculative decoding requires a shared
  vocabulary (the draft proposes token IDs the target scores). `new_with_opts` now
  rejects a draft whose vocab differs from the target's with a clear error (instead
  of emitting garbage), and `with_auto_draft` picks a draft from the **same family**
  as the target (Qwen→qwen2.5-0.5b, SmolLM2→smollm2-135m).

Notes/limits: best for single-user decode-bound serving (2–3×). Requires Llama/Phi
CPU engines (the cache-rollback path); MLX has no incremental cache rollback so it
isn't used for speculative. Future: NGram/prompt-lookup drafting (no draft model).

## Deferred (designed, not yet implemented)

### Phase 5 — Continuous batching + PagedAttention
The forward engine is **strictly single-sequence**: `forward_logits(&[u32])` has no
batch dimension and the KV cache is `[1, n_kv, max_seq, head_dim]` with one `seq_len`
per layer. True continuous (in-flight) batching and PagedAttention require:
1. A **batched multi-sequence forward** (per-sequence positions/masks in one step).
2. A **block-pool KV cache** keyed by sequence id (fixed-size blocks from a central
   free list — vLLM 16-tok, mistral.rs 32-tok), enabling eviction + prefix sharing.
3. A **scheduler** mixing prefill+decode each step with a per-step token budget
   (chunked prefill) and parallel slots.
This is a large, high-risk engine rewrite (the released engine is single-sequence).
**mistral.rs** is the pure-Rust precedent (PagedAttention + default continuous
batching on CPU/CUDA/Metal). Recommended sequencing: block-pool KV cache → batched
forward → scheduler → PagedAttention kernels. Until then, concurrency is handled by
per-model serialization + the admission semaphore (Phase 3).
