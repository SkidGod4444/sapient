//! Integration test: Inference engine testing (KV cache and correctness).

use sapient_core::{DType, Shape, Tensor};
use sapient_models::forward::common::update_kv_cache;

#[test]
fn test_update_kv_cache() {
    let b = 1;
    let n_kv = 2;
    let max_seq = 4;
    let hd = 2;

    // Cache is initialized with zeros
    let mut cache = Tensor::zeros(Shape::new([b, n_kv, max_seq, hd]), DType::F32).unwrap();

    // First insertion: seq_len = 1
    let k1_data = vec![
        1.0, 1.0, // head 0
        2.0, 2.0, // head 1
    ];
    let k1 = Tensor::from_f32(&k1_data, Shape::new([b, n_kv, 1, hd])).unwrap();

    let view1 = update_kv_cache(&mut cache, 0, &k1).unwrap();
    assert_eq!(view1.shape().dims(), &[1, 2, 1, 2]);
    drop(view1);

    // Second insertion: seq_len = 2
    let k2_data = vec![
        3.0, 3.0, // head 0, pos 2
        4.0, 4.0, // head 0, pos 3
        5.0, 5.0, // head 1, pos 2
        6.0, 6.0, // head 1, pos 3
    ];
    let k2 = Tensor::from_f32(&k2_data, Shape::new([b, n_kv, 2, hd])).unwrap();

    let view2 = update_kv_cache(&mut cache, 1, &k2).unwrap();
    assert_eq!(view2.shape().dims(), &[1, 2, 3, 2]);
    drop(view2);

    // Check full state
    let cache_slice = cache.as_f32_slice();
    // Cache has max_seq=4
    // Head 0:
    // pos 0: 1, 1
    // pos 1: 3, 3
    // pos 2: 4, 4
    // pos 3: 0, 0
    // Head 1:
    // pos 0: 2, 2
    // pos 1: 5, 5
    // pos 2: 6, 6
    // pos 3: 0, 0

    let h0 = &cache_slice[0..8];
    assert_eq!(h0, &[1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 0.0, 0.0]);

    let h1 = &cache_slice[8..16];
    assert_eq!(h1, &[2.0, 2.0, 5.0, 5.0, 6.0, 6.0, 0.0, 0.0]);
}
