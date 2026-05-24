//! Integration test: MLP graph end-to-end on CPU backend.

use std::collections::HashMap;

use sapient_core::{DType, Tensor};
use sapient_ir::{Graph, OpType};
use sapient_backends_cpu::backend::{CpuBackend, ExecutionBackend};

/// Build a 2-layer MLP: x (2×4) → MatMul+bias → Relu → MatMul+bias → output
fn build_mlp_graph() -> Graph {
    let mut g = Graph::new("mlp_test");

    // Inputs.
    let x = g.add_input("x", Some(sapient_core::Shape::new([2, 4])), Some(DType::F32));

    // Layer 1 weights [4,8] and bias [8].
    let w1 = g.add_constant(
        Tensor::from_f32(&vec![0.1f32; 32], vec![4, 8]).unwrap(),
        Some("w1".into()),
    );
    let b1 = g.add_constant(
        Tensor::from_f32(&vec![0.01f32; 8], vec![8]).unwrap(),
        Some("b1".into()),
    );

    // Layer 2 weights [8,2] and bias [2].
    let w2 = g.add_constant(
        Tensor::from_f32(&vec![0.1f32; 16], vec![8, 2]).unwrap(),
        Some("w2".into()),
    );
    let b2 = g.add_constant(
        Tensor::from_f32(&vec![0.0f32; 2], vec![2]).unwrap(),
        Some("b2".into()),
    );

    // Layer 1: x @ w1 + b1.
    let mm1  = g.add_op(OpType::MatMul, vec![x, w1], 1, Some("mm1".into()));
    let add1 = g.add_op(OpType::Add, vec![mm1, b1], 1, Some("add1".into()));
    let relu = g.add_op(OpType::Relu, vec![add1], 1, Some("relu".into()));

    // Layer 2: relu @ w2 + b2.
    let mm2  = g.add_op(OpType::MatMul, vec![relu, w2], 1, Some("mm2".into()));
    let add2 = g.add_op(OpType::Add, vec![mm2, b2], 1, Some("add2".into()));

    g.mark_output(add2, "logits");
    g
}

#[test]
fn mlp_end_to_end() {
    let graph = build_mlp_graph();
    graph.validate().expect("graph should be valid");

    let backend = CpuBackend::default();

    let x_data = vec![1.0f32; 8]; // 2×4
    let inputs = HashMap::from([
        ("x".to_owned(), Tensor::from_f32(&x_data, vec![2, 4]).unwrap()),
    ]);

    let outputs = backend.execute(&graph, inputs).expect("execution failed");
    assert_eq!(outputs.len(), 1);

    let out = &outputs[0];
    assert_eq!(out.shape().dims(), &[2, 2], "output shape should be [2,2]");
    assert!(out.as_f32_slice().iter().all(|v| v.is_finite()), "all outputs must be finite");
}

#[test]
fn softmax_after_mlp() {
    let mut graph = build_mlp_graph();

    // Append softmax after the output.
    let logits_id = graph.outputs[0];
    // Find source.
    use sapient_ir::node::Node;
    let source_id = if let Some(Node::Output { source, .. }) = graph.get(logits_id) {
        *source
    } else {
        panic!("expected Output node");
    };

    graph.outputs.clear();
    let sm = graph.add_op(
        OpType::Softmax { axis: -1 },
        vec![source_id],
        1,
        Some("softmax".into()),
    );
    graph.mark_output(sm, "probs");

    let backend = CpuBackend::default();
    let x_data = vec![1.0f32; 8];
    let inputs = HashMap::from([
        ("x".to_owned(), Tensor::from_f32(&x_data, vec![2, 4]).unwrap()),
    ]);

    let outputs = backend.execute(&graph, inputs).unwrap();
    let probs = outputs[0].as_f32_slice();
    // Each row should sum to 1.
    for row in probs.chunks(2) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "prob row should sum to 1, got {sum}");
    }
}

#[test]
fn batching_throughput() {
    use sapient_scheduler::{DynamicBatchScheduler, Request, StaticBatchScheduler};
    use std::time::Duration;

    // Verify the dynamic scheduler batches within the timeout window.
    let mut sched = DynamicBatchScheduler::new(8, Duration::from_millis(1));
    for _ in 0..3 {
        sched.submit(Request::new(HashMap::new()));
    }
    std::thread::sleep(Duration::from_millis(5));
    let batch = sched.try_form_batch().expect("should have formed a batch");
    assert_eq!(batch.len(), 3);
}
