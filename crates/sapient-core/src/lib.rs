// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! SAPIENT core — re-exports for all primary types.

pub mod buffer;
pub mod dtype;
pub mod error;
pub mod shape;
pub mod tensor;

pub use buffer::{Buffer, BufferHandle, CpuBuffer};
pub use dtype::{
    DType, K_QUANT_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QUANT_BLOCK_SIZE,
};
pub use error::{Result, SapientError};
pub use shape::Shape;
pub use tensor::Tensor;
