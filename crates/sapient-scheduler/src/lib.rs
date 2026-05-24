#![allow(
    unused_imports,
    unused_variables,
    clippy::implied_bounds_in_impls,
    clippy::unnecessary_map_or
)]

//! SAPIENT batching scheduler and async executor.

pub mod batcher;
pub mod executor;
pub mod request;
pub mod scheduler;

pub use batcher::Batcher;
pub use executor::Executor;
pub use request::{Request, Response};
pub use scheduler::{BatchScheduler, DynamicBatchScheduler, StaticBatchScheduler};
