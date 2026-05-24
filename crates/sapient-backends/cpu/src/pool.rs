//! Tensor memory pool with LRU eviction.
//!
//! The pool reduces allocation pressure for intermediate tensors that are
//! created and freed on every forward pass.

use std::collections::HashMap;

use parking_lot::Mutex;

use sapient_core::buffer::BufferHandle;
use sapient_core::DType;

// ── Entry ────────────────────────────────────────────────────────────────────

struct PoolEntry {
    handle: BufferHandle,
    last_used: std::time::Instant,
    capacity: usize,
}

// ── PoolAllocator ─────────────────────────────────────────────────────────────

/// LRU memory pool for CPU tensor buffers.
///
/// Buffers are keyed by `(numel, dtype)`.  When a caller returns a buffer, it
/// can be re-acquired on the next allocation of the same size, avoiding heap
/// `malloc`/`free` on the hot path.
pub struct PoolAllocator {
    inner: Mutex<PoolInner>,
}

struct PoolInner {
    /// Available buffers, grouped by byte capacity.
    free: HashMap<usize, Vec<PoolEntry>>,
    /// Total bytes currently held in the pool.
    used_bytes: usize,
    /// Maximum bytes the pool will hold.
    capacity: usize,
}

impl PoolAllocator {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(PoolInner {
                free: HashMap::new(),
                used_bytes: 0,
                capacity: capacity_bytes,
            }),
        }
    }

    /// Try to acquire a buffer for `numel` elements of `dtype`.
    /// Returns `None` if the pool has no suitable entry — caller should
    /// allocate fresh.
    pub fn acquire(&self, numel: usize, dtype: DType) -> Option<BufferHandle> {
        let byte_size = dtype.byte_count(numel);
        let mut inner = self.inner.lock();

        // Look for an exact or larger buffer.
        if let Some(entries) = inner.free.get_mut(&byte_size) {
            if let Some(entry) = entries.pop() {
                inner.used_bytes = inner.used_bytes.saturating_sub(entry.capacity);
                return Some(entry.handle);
            }
        }
        None
    }

    /// Return a buffer to the pool after use.
    ///
    /// If the pool is over capacity, evict the least-recently-used entries.
    pub fn release(&self, handle: BufferHandle, numel: usize, dtype: DType) {
        let byte_size = dtype.byte_count(numel);
        let mut inner = self.inner.lock();

        // Evict LRU entries if needed.
        while inner.used_bytes + byte_size > inner.capacity {
            if !Self::evict_lru(&mut inner) {
                break;
            }
        }

        if inner.used_bytes + byte_size <= inner.capacity {
            inner.used_bytes += byte_size;
            inner.free.entry(byte_size).or_default().push(PoolEntry {
                handle,
                last_used: std::time::Instant::now(),
                capacity: byte_size,
            });
        }
        // If still over capacity: discard (buffer drops).
    }

    fn evict_lru(inner: &mut PoolInner) -> bool {
        // Find the oldest entry across all buckets.
        let mut oldest_key: Option<usize> = None;
        let mut oldest_time = std::time::Instant::now();

        for (&key, entries) in &inner.free {
            for entry in entries {
                if entry.last_used < oldest_time {
                    oldest_time = entry.last_used;
                    oldest_key = Some(key);
                }
            }
        }

        if let Some(key) = oldest_key {
            if let Some(entries) = inner.free.get_mut(&key) {
                if let Some(entry) = entries.pop() {
                    inner.used_bytes = inner.used_bytes.saturating_sub(entry.capacity);
                    return true;
                }
            }
        }
        false
    }

    /// Total bytes currently pooled.
    pub fn used_bytes(&self) -> usize {
        self.inner.lock().used_bytes
    }

    /// Pool capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.inner.lock().capacity
    }
}

impl std::fmt::Debug for PoolAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("PoolAllocator")
            .field("used_bytes", &inner.used_bytes)
            .field("capacity", &inner.capacity)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sapient_core::buffer::CpuBuffer;

    #[test]
    fn acquire_release_cycle() {
        let pool = PoolAllocator::new(1024 * 1024);
        // Nothing in pool → None.
        assert!(pool.acquire(16, DType::F32).is_none());

        // Put something in.
        let buf = BufferHandle::new(CpuBuffer::zeros(16, DType::F32).unwrap());
        pool.release(buf, 16, DType::F32);
        assert_eq!(pool.used_bytes(), 64); // 16 * 4 bytes

        // Now acquire should succeed.
        let h = pool.acquire(16, DType::F32);
        assert!(h.is_some());
        assert_eq!(pool.used_bytes(), 0);
    }
}
