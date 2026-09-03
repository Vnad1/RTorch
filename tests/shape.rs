// Shape / index / grad-length misuse must be caught explicitly (clear panic),
// not silently produce wrong values. These guard the "runs but is wrong" class
// of bugs where incompatible shapes used to wrap around via max+modulo.
use rtorch::tensor::{self, Tensor};
use rtorch::autograd::{self, Adam};

#[test]
fn matmul_k_mismatch_panics() {
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_data(vec![1.0, 2.0, 3.0], 3, 1); // a.c=2 vs b.r=3
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tensor::matmul(&a, &b)));
    assert!(res.is_err(), "matmul with a.c != b.r must error, not silently compute");
}

#[test]
fn add_incompatible_shapes_panic() {
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2); // rows 2 vs 3 -> not broadcastable
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tensor::add(&a, &b)));
    assert!(res.is_err(), "add 2×2 with 3×2 must error (was silently wrapping)");
}

#[test]
fn add_broadcast_compatible_still_works() {
    // 1×c and r×1 broadcasts remain valid (real broadcasting).
    let a = Tensor::from_data(vec![1.0, 2.0], 1, 2);
    let b = Tensor::from_data(vec![10.0, 20.0, 30.0], 3, 1);
    let c = tensor::add(&a, &b);
    assert_eq!((c.r, c.c), (3, 2));
    // c[i,j] = a[0,j] + b[i,0]
    assert_eq!(c.data, vec![11.0, 12.0, 21.0, 22.0, 31.0, 32.0]);
}

#[test]
fn gather_out_of_range_panics() {
    let e = autograd::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2); // 2 rows
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| autograd::gather(&e, 5)));
    assert!(res.is_err(), "gather index 5 on size-2 embedding must error (was clamping)");
}

#[test]
fn adam_grad_len_mismatch_panics() {
    let p = autograd::from_data(vec![1.0, 2.0], 1, 2);
    p.borrow_mut().grad = vec![0.1]; // wrong length
    let mut opt = Adam::new(0.1);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| opt.step(&[p])));
    assert!(res.is_err(), "adam with mismatched grad length must error");
}

#[test]
fn sgd_grad_len_mismatch_panics() {
    let mut p = Tensor::from_data(vec![1.0, 2.0], 1, 2);
    p.grad = vec![0.1, 0.2, 0.3]; // wrong length
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tensor::sgd_step(&mut [&mut p], 0.1)));
    assert!(res.is_err(), "sgd with mismatched grad length must error");
}
