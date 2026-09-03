// Model-framework cycle (the "Striker depends on RTorch" contract).
// A model layer built on RTorch does: forward → autograd loss → backward → Adam
// → save weights+optimizer to an .rtw checkpoint → reload → continue training.
// This is the strongest available proof (no standalone striker crate in the
// workspace depends on rtorch) that RTorch is a reliable compute base.
use rtorch::autograd::{self, Adam};
use rtorch::rtw;

fn set_pred_grad_mse(pred: &autograd::Var, target: &[f64]) {
    let mut b = pred.borrow_mut();
    for i in 0..b.data.len() {
        let t = target.get(i).copied().unwrap_or(0.0);
        b.grad[i] = 2.0 * (b.data[i] - t);
    }
}
fn mse(c: &[f64], t: &[f64]) -> f64 {
    c.iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / c.len() as f64
}

// Train one Adam step on pred = x·w + b -> target; returns the loss.
fn step(
    x: &autograd::Var,
    target: &[f64],
    w: &autograd::Var,
    b: &autograd::Var,
    opt: &mut Adam,
) -> f64 {
    let pred = autograd::add(&autograd::matmul(x, w), b);
    let p = pred.borrow().data.clone();
    set_pred_grad_mse(&pred, target);
    autograd::backward(&pred);
    opt.step(&[w.clone(), b.clone()]);
    w.borrow_mut().grad.iter_mut().for_each(|g| *g = 0.0);
    b.borrow_mut().grad.iter_mut().for_each(|g| *g = 0.0);
    mse(&p, target)
}

#[test]
fn train_save_load_continue_full_cycle() {
    // Fit pred = x·w + b to target = 2x, then checkpoint + reload + continue.
    let x = autograd::from_data(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
    let target = vec![2.0, 4.0, 6.0, 8.0];
    let w = autograd::from_data(vec![0.0], 1, 1);
    let b = autograd::from_data(vec![0.0], 1, 1);
    let mut opt = Adam::new(0.05);

    for _ in 0..300 {
        step(&x, &target, &w, &b, &mut opt);
    }
    let before = mse(
        &{
            let pred = autograd::add(&autograd::matmul(&x, &w), &b);
            pred.borrow().data.clone()
        },
        &target,
    );

    // Serialize weights + optimizer state to an RTW model checkpoint.
    let (ms, vs, t) = opt.state(&[w.clone(), b.clone()]);
    let f32s = |v: &Vec<Vec<f64>>| {
        v.iter()
            .map(|l| l.iter().map(|&x| x as f32).collect::<Vec<f32>>())
            .collect::<Vec<Vec<f32>>>()
    };
    let model = rtw::Model {
        name: "demo-net".into(),
        version: 1,
        params: vec![
            rtw::NamedTensor {
                name: "W".into(),
                shape: vec![1, 1],
                dtype: rtw::DTYPE_FP32,
                data: w.borrow().data.iter().map(|&f| f as f32).collect(),
            },
            rtw::NamedTensor {
                name: "b".into(),
                shape: vec![1, 1],
                dtype: rtw::DTYPE_FP32,
                data: b.borrow().data.iter().map(|&f| f as f32).collect(),
            },
        ],
        opt: Some(rtw::OptState {
            m: f32s(&ms),
            v: f32s(&vs),
            t,
        }),
    };
    let bytes = rtw::encode(&rtw::model_rtw(&model));
    let dec = rtw::decode(&bytes).unwrap();
    assert_eq!(dec.kind, rtw::KIND_MODEL);
    let restored = rtw::decode_model(&dec.data).unwrap();
    assert_eq!(restored.params.len(), 2);
    assert_eq!(restored.params[0].name, "W");

    // Rebuild weights from the checkpoint and continue training; loss must fall.
    let w2 = autograd::from_data(
        restored.params[0].data.iter().map(|&f| f as f64).collect(),
        1,
        1,
    );
    let b2 = autograd::from_data(
        restored.params[1].data.iter().map(|&f| f as f64).collect(),
        1,
        1,
    );
    let mut opt2 = Adam::new(0.05);
    if let Some(o) = restored.opt {
        let tof = |v: &Vec<Vec<f32>>| {
            v.iter()
                .map(|l| l.iter().map(|&x| x as f64).collect::<Vec<f64>>())
                .collect::<Vec<Vec<f64>>>()
        };
        opt2.load_state(&[w2.clone(), b2.clone()], &tof(&o.m), &tof(&o.v), o.t);
    }
    let mut last = mse(
        &{
            let pred = autograd::add(&autograd::matmul(&x, &w2), &b2);
            pred.borrow().data.clone()
        },
        &target,
    );
    for _ in 0..200 {
        last = step(&x, &target, &w2, &b2, &mut opt2);
    }

    assert!(
        last < before.max(1.0),
        "reload + continue not improving: {before} -> {last}"
    );
    assert!(
        last < 1e-2,
        "model did not converge after checkpoint reload: mse {last}"
    );
    // w approaches ~2 (we fit 2x).
    assert!(
        (w2.borrow().data[0] - 2.0).abs() < 0.2,
        "W={} expected ~2",
        w2.borrow().data[0]
    );
}
