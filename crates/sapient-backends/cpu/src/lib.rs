#![allow(
    clippy::get_first,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::needless_borrows_for_generic_args
)]

//! CPU execution backend for SAPIENT.

pub mod backend;
pub mod kernels;
pub mod pool;

pub use backend::CpuBackend;
pub use pool::PoolAllocator;
