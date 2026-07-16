// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

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
pub mod spinpool;
pub mod thermal;

pub use backend::CpuBackend;
pub use pool::PoolAllocator;
