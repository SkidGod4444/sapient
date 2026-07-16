// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Criterion benchmarks for CPU ops.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sapient_backends_cpu::kernels::{layernorm, matmul, softmax};
use sapient_core::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for size in [64usize, 128, 256, 512, 1024] {
        let a = Tensor::from_f32(&vec![1.0f32; size * size], vec![size, size]).unwrap();
        let b = Tensor::from_f32(&vec![1.0f32; size * size], vec![size, size]).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, _| {
            bencher.iter(|| black_box(matmul::matmul(&a, &b).unwrap()));
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    for seq_len in [128usize, 512, 2048] {
        let x = Tensor::from_f32(&vec![0.1f32; seq_len], vec![1, seq_len]).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(seq_len),
            &seq_len,
            |bencher, _| {
                bencher.iter(|| black_box(softmax::softmax(&x, 1).unwrap()));
            },
        );
    }
    group.finish();
}

fn bench_layernorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm");
    for hidden in [512usize, 1024, 4096] {
        let x = Tensor::from_f32(&vec![0.5f32; 8 * hidden], vec![8, hidden]).unwrap();
        let w = Tensor::from_f32(&vec![1.0f32; hidden], vec![hidden]).unwrap();
        let b = Tensor::from_f32(&vec![0.0f32; hidden], vec![hidden]).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(hidden),
            &hidden,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(layernorm::layer_norm(&x, Some(&w), Some(&b), -1, 1e-5).unwrap())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_softmax, bench_layernorm);
criterion_main!(benches);
