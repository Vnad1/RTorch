// Tensor correctness — CPU reference ops (rtorch::tensor).
use rtorch::tensor::{self, Tensor};

#[test]
fn matmul_2x2() {
    // [[1,2],[3,4]] . [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = tensor::matmul(&a, &b);
    assert_eq!(c.r, 2);
    assert_eq!(c.c, 2);
    let expect = [19.0, 22.0, 43.0, 50.0];
    for (i, e) in expect.iter().enumerate() {
        assert!((c.data[i] - e).abs() < 1e-12, "matmul[{}]={} expected {}", i, c.data[i], e);
    }
}

#[test]
fn matmul_dims() {
    // A[3x4] . B[4x2] => C[3x2]
    let a = Tensor::from_data((0..12).map(|i| i as f64).collect(), 3, 4);
    let b = Tensor::from_data((0..8).map(|i| (i as f64) * 0.5).collect(), 4, 2);
    let c = tensor::matmul(&a, &b);
    assert_eq!((c.r, c.c), (3, 2));
    assert_eq!(c.len(), 6);
}

#[test]
fn add_broadcast() {
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_data(vec![10.0, 20.0], 1, 2); // 1x2 broadcast over rows
    let c = tensor::add(&a, &b);
    assert_eq!(c.data, vec![11.0, 22.0, 13.0, 24.0]);
}

#[test]
fn tanh_matches_std() {
    let x = Tensor::from_data(vec![-1.0, 0.0, 0.5, 2.0], 2, 2);
    let y = tensor::tanh(&x);
    for i in 0..4 {
        assert!((y.data[i] - x.data[i].tanh()).abs() < 1e-12);
    }
}

#[test]
fn softmax_rows_sum_to_one() {
    let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let p = tensor::softmax_row(&x);
    for r in 0..2 {
        let s: f64 = (0..3).map(|j| p.data[r * 3 + j]).sum();
        assert!((s - 1.0).abs() < 1e-12, "row {r} sums to {s}");
    }
    // monotonic: softmax of [1,2,3] is increasing
    assert!(p.data[0] < p.data[1] && p.data[1] < p.data[2]);
}

#[test]
fn scal_and_onehot() {
    let x = Tensor::from_data(vec![1.0, 2.0, 3.0], 1, 3);
    let y = tensor::scal(2.0, &x);
    assert_eq!(y.data, vec![2.0, 4.0, 6.0]);
    let oh = tensor::onehot(2, 4);
    assert_eq!(oh.data, vec![0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn sgd_step_updates() {
    let mut p = Tensor::from_data(vec![1.0, -2.0], 1, 2);
    p.grad = vec![0.1, 0.5];
    tensor::sgd_step(&mut [&mut p], 0.1);
    assert!((p.data[0] - 0.99).abs() < 1e-12);
    assert!((p.data[1] - (-2.05)).abs() < 1e-12);
}
