// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! CPU kernels — one sub-module per family.

pub mod attention;
pub mod conv2d;
pub mod elementwise;
pub mod layernorm;
pub mod matmul;
pub mod quant;
pub mod reduce;
pub mod rope;
pub mod softmax;
