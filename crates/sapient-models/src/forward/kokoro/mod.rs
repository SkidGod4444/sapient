// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Kokoro-82M text-to-speech (StyleTTS2 + ISTFTNet) — a pure-Rust forward pass.
//!
//! Kokoro is a *non-autoregressive* TTS: a single forward pass turns a phoneme
//! sequence + a style vector into a 24 kHz waveform, with no codec-token decode
//! loop. That sidesteps the tokens-per-second ceiling that bottlenecks the
//! autoregressive Orpheus-3B + SNAC path (~0.18× real-time), which is why it is
//! the real-time TTS option for `sapient converse`.
//!
//! Pipeline (module names match the checkpoint key prefixes):
//! `phonemes → bert (ALBERT) → bert_encoder → predictor (durations + F0 + N)
//!  → length-regulate → text_encoder → decoder (ISTFTNet) → waveform`.
//!
//! This file is being built up phase by phase; [`ops`] holds the new CPU
//! primitives Kokoro needs that the rest of SAPIENT did not already provide.

// Built up phase by phase; primitives land before their call sites are wired.
// This allowance is removed once the engine is assembled and consumed.
#![allow(dead_code)]

mod albert;
mod decoder;
mod loader;
mod model;
mod ops;
mod predictor;

pub use loader::KokoroConfig;
pub use model::{DecoderStreamInputs, KokoroModel, KOKORO_SAMPLE_RATE};

#[cfg(test)]
mod stage_tests;
