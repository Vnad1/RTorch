// Tensor Shape/DType abstraction — dtype taxonomy, shape/strides/numel,
// reshape/view/transpose, and multi-dim (e.g. [B,T,D]) construction.
use rtorch::tensor::{self, DType, Shape, Tensor};

#[test]
fn dtype_name_and_size() {
    assert_eq!(DType::F32.name(), "f32");
    assert_eq!(DType::F64.size(), 8);
    assert_eq!(DType::F16.size(), 2);
    assert_eq!(DType::BF16.size(), 2);
    assert_eq!(DType::I32.size(), 4);
}

#[test]
fn shape_numel_rank_strides() {
    let s = Shape::new(vec![2, 3, 4]); // [B,T,D]-style
    assert_eq!(s.rank(), 3);
    assert_eq!(s.numel(), 24);
    assert_eq!(s.dim(1), 3);
    // row-major contiguous strides: [12, 4, 1]
    assert_eq!(s.strides(), vec![12, 4, 1]);
}

#[test]
fn from_shape_multi_dim_projects_2d() {
    // A [B=2, T=3, D=4] tensor stored contiguous; r=B, c=T*D for the 2D op path.
    let t = Tensor::from_shape(
        (0..24).map(|i| i as f64).collect(),
        vec![2, 3, 4],
        DType::F64,
    )
    .unwrap();
    assert_eq!(t.shape(), Shape::new(vec![2, 3, 4]));
    assert_eq!(t.numel(), 24);
    assert_eq!((t.r, t.c), (2, 12)); // 2D projection: [2, 12]
    assert_eq!(t.data[5], 5.0); // linear layout preserved
}

#[test]
fn reshape_and_view_match_numel() {
    let t = Tensor::from_data((0..6).map(|i| i as f64).collect(), 2, 3);
    let r = t.reshape(vec![3, 2]).unwrap();
    assert_eq!((r.r, r.c), (3, 2));
    assert_eq!(r.dims, vec![3, 2]);
    let v = t.view(vec![6, 1]).unwrap();
    assert_eq!((v.r, v.c), (6, 1));
    assert!(
        t.reshape(vec![4, 2]).is_err(),
        "reshape numel mismatch must error"
    );
}

#[test]
fn transpose_swaps_2d() {
    let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let tt = t.transpose();
    assert_eq!((tt.r, tt.c), (3, 2));
    // t = [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    assert_eq!(tt.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn dtype_mismatch_add_panics() {
    let a = Tensor::from_data(vec![1.0], 1, 1).as_dtype(DType::F16);
    let b = Tensor::from_data(vec![2.0], 1, 1).as_dtype(DType::F64);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tensor::add(&a, &b)));
    assert!(
        r.is_err(),
        "add with differing dtype must error, not silently promote"
    );
}

#[test]
fn matmul_preserves_broadcast_and_2d_ops() {
    // Existing 2D ops still work and keep dtype.
    let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = tensor::matmul(&a, &b);
    assert_eq!(c.dtype, DType::F64);
    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]); // [[1,2],[3,4]]·[[5,6],[7,8]]
}
