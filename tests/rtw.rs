// RTW (.rtw) container encode/decode roundtrip (rtorch::rtw).
use rtorch::rtw;

fn f32_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

#[test]
fn result_fp32_roundtrip() {
    let data = f32_bytes(&[1.0, -2.5, 3.75, 0.0, 1e-6, 42.0]);
    let rtw = rtw::Rtw { kind: rtw::KIND_RESULT, dtype: rtw::DTYPE_FP32, shape: vec![2, 3], data, kernel: None };
    let bytes = rtw::encode(&rtw);
    let dec = rtw::decode(&bytes).expect("decode");
    assert_eq!(dec.kind, rtw::KIND_RESULT);
    assert_eq!(dec.dtype, rtw::DTYPE_FP32);
    assert_eq!(dec.shape, vec![2, 3]);
    assert_eq!(dec.data, rtw.data);
    assert_eq!(dec.kernel, None);
    assert_eq!(dec.count(), 6);
}

#[test]
fn kernel_roundtrip_keeps_source() {
    let src = b"unsigned long long rtorch_output_size(...){...}".to_vec();
    let rtw = rtw::Rtw { kind: rtw::KIND_KERNEL, dtype: rtw::DTYPE_BYTES, shape: vec![], data: vec![], kernel: Some(src.clone()) };
    let bytes = rtw::encode(&rtw);
    let dec = rtw::decode(&bytes).expect("decode");
    assert_eq!(dec.kind, rtw::KIND_KERNEL);
    assert_eq!(dec.kernel, Some(src));
}

#[test]
fn model_kind_roundtrip() {
    let data = f32_bytes(&[0.1, 0.2, 0.3, 0.4]);
    let rtw = rtw::Rtw { kind: rtw::KIND_MODEL, dtype: rtw::DTYPE_FP32, shape: vec![4], data, kernel: None };
    let bytes = rtw::encode(&rtw);
    let dec = rtw::decode(&bytes).expect("decode");
    assert_eq!(dec.kind, rtw::KIND_MODEL);
    assert_eq!(dec.shape, vec![4]);
    assert_eq!(dec.count(), 4);
}

#[test]
fn dtype_width_and_name() {
    assert_eq!(rtw::dtype_width(rtw::DTYPE_FP32), 4);
    assert_eq!(rtw::dtype_width(rtw::DTYPE_FP16), 2);
    assert_eq!(rtw::dtype_width(rtw::DTYPE_FP8), 1);
    assert_eq!(rtw::dtype_width(rtw::DTYPE_INT32), 4);
    assert_eq!(rtw::dtype_name(rtw::DTYPE_FP32), "fp32");
}

#[test]
fn rejects_bad_magic() {
    let bad = b"NOTRTW............".to_vec();
    assert!(rtw::decode(&bad).is_err());
}

#[test]
fn rejects_truncated() {
    let data = f32_bytes(&[1.0, 2.0]);
    let rtw = rtw::Rtw { kind: rtw::KIND_RESULT, dtype: rtw::DTYPE_FP32, shape: vec![2], data, kernel: None };
    let bytes = rtw::encode(&rtw);
    // Truncate deep into the payload region: data_len is already read, but the
    // declared data region is now short -> decode returns an error, not a panic.
    let cut = bytes.len() / 2;
    let mut t = bytes[..cut].to_vec();
    assert!(rtw::decode(&t).is_err());
    // Even a bare header must return Err (not panic).
    t = bytes[..8].to_vec();
    assert!(rtw::decode(&t).is_err());
}
