//! CPU execution backend for SAPIENT.

pub mod backend;
pub mod kernels;
pub mod pool;

pub use backend::CpuBackend;
pub use pool::PoolAllocator;
