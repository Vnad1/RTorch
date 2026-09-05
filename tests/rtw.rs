// RTW (.rtw) container encode/decode roundtrip (rtorch::rtw).
use rtorch::rtw;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn result_fp32_roundtrip() {
    let data = f32_bytes(&[1.0, -2.5, 3.75, 0.0, 1e-6, 42.0]);
    let rtw = rtw::Rtw {
        kind: rtw::KIND_RESULT,
        dtype: rtw::DTYPE_FP32,
        shape: vec![2, 3],
        data,
        kernel: None,
    };
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
    let rtw = rtw::Rtw {
        kind: rtw::KIND_KERNEL,
        dtype: rtw::DTYPE_BYTES,
        shape: vec![],
        data: vec![],
        kernel: Some(src.clone()),
    };
    let bytes = rtw::encode(&rtw);
    let dec = rtw::decode(&bytes).expect("decode");
    assert_eq!(dec.kind, rtw::KIND_KERNEL);
    assert_eq!(dec.kernel, Some(src));
}

#[test]
fn model_kind_roundtrip() {
    let data = f32_bytes(&[0.1, 0.2, 0.3, 0.4]);
    let rtw = rtw::Rtw {
        kind: rtw::KIND_MODEL,
        dtype: rtw::DTYPE_FP32,
        shape: vec![4],
        data,
        kernel: None,
    };
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
    let rtw = rtw::Rtw {
        kind: rtw::KIND_RESULT,
        dtype: rtw::DTYPE_FP32,
        shape: vec![2],
        data,
        kernel: None,
    };
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

#[test]
fn model_roundtrip_params_and_opt() {
    let m = rtw::Model {
        name: "test-net".into(),
        version: 3,
        params: vec![
            rtw::NamedTensor {
                name: "W".into(),
                shape: vec![2, 3],
                dtype: rtw::DTYPE_FP32,
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
            rtw::NamedTensor {
                name: "b".into(),
                shape: vec![3],
                dtype: rtw::DTYPE_FP32,
                data: vec![0.1, 0.2, 0.3],
            },
        ],
        opt: Some(rtw::OptState {
            m: vec![vec![1.0, 2.0], vec![3.0]],
            v: vec![vec![0.5], vec![0.6, 0.7, 0.8]],
            t: 42,
        }),
    };
    let bytes = rtw::encode_model(&m);
    let d = rtw::decode_model(&bytes).expect("decode model");
    assert_eq!(d.name, "test-net");
    assert_eq!(d.version, 3);
    assert_eq!(d.params.len(), 2);
    assert_eq!(d.params[0].name, "W");
    assert_eq!(d.params[0].shape, vec![2, 3]);
    assert_eq!(d.params[0].data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(d.params[1].data, vec![0.1, 0.2, 0.3]);
    let o = d.opt.unwrap();
    assert_eq!(o.t, 42);
    assert_eq!(o.m, vec![vec![1.0, 2.0], vec![3.0]]);
    assert_eq!(o.v, vec![vec![0.5], vec![0.6, 0.7, 0.8]]);
}

#[test]
fn model_in_rtw_container_roundtrips() {
    let m = rtw::Model {
        name: "checkpoint".into(),
        version: 1,
        params: vec![rtw::NamedTensor {
            name: "A".into(),
            shape: vec![2, 2],
            dtype: rtw::DTYPE_FP32,
            data: vec![1.0, 0.0, 0.0, 1.0],
        }],
        opt: None,
    };
    let rtw = rtw::model_rtw(&m);
    assert_eq!(rtw.kind, rtw::KIND_MODEL);
    let bytes = rtw::encode(&rtw);
    let dec = rtw::decode(&bytes).expect("decode container");
    assert_eq!(dec.kind, rtw::KIND_MODEL);
    let dm = rtw::decode_model(&dec.data).expect("decode model payload");
    assert_eq!(dm.name, "checkpoint");
    assert_eq!(dm.params[0].data, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn memory_roundtrip_fragments() {
    let mem = rtw::Memory {
        fragments: vec![
            rtw::MemoryFragment {
                id: 7,
                state: vec![0.1, 0.2, 0.3],
                strength: 0.9,
            },
            rtw::MemoryFragment {
                id: 9,
                state: vec![-1.0, 2.5],
                strength: 0.4,
            },
        ],
    };
    let bytes = rtw::encode_memory(&mem);
    let d = rtw::decode_memory(&bytes).expect("decode memory");
    assert_eq!(d.fragments.len(), 2);
    assert_eq!(d.fragments[0].id, 7);
    assert_eq!(d.fragments[0].state, vec![0.1, 0.2, 0.3]);
    assert!((d.fragments[0].strength - 0.9).abs() < 1e-9);
    assert_eq!(d.fragments[1].state, vec![-1.0, 2.5]);

    let rtww = rtw::memory_rtw(&mem);
    let bytes2 = rtw::encode(&rtww);
    let dec = rtw::decode(&bytes2).expect("decode memory container");
    assert_eq!(dec.kind, rtw::KIND_MEMORY);
    let dm = rtw::decode_memory(&dec.data).expect("decode memory payload");
    assert_eq!(dm.fragments.len(), 2);
}

#[test]
fn truncated_memory_returns_err() {
    // A memory payload truncated mid-way must return Err (not panic/abort).
    let mem = rtw::Memory {
        fragments: vec![
            rtw::MemoryFragment { id: 1, state: vec![0.1, 0.2], strength: 0.5 },
            rtw::MemoryFragment { id: 2, state: vec![0.3], strength: 0.7 },
        ],
    };
    let mut bytes = rtw::encode_memory(&mem);
    // Chop the last 6 bytes (strength + state tail) so the count no longer holds.
    bytes.truncate(bytes.len() - 6);
    let r = rtw::decode_memory(&bytes);
    assert!(r.is_err(), "truncated memory must be an error");
}

#[test]
fn hostile_count_returns_err_not_abort() {
    // A payload whose leading count is huge (0xFFFFFFFF) must return Err, not
    // drive a multi-GB `Vec::with_capacity` allocation (which would abort the
    // process). This is the regression test for `bound_capacity` in
    // decode_memory / decode_model / decode.
    let hostile = [0xFFu8; 16];
    assert!(rtw::decode_memory(&hostile).is_err());
    assert!(rtw::decode_model(&hostile).is_err());
    assert!(rtw::decode(&hostile).is_err());
}
