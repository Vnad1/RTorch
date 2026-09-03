// Training-step + optimizer checkpoint-reload correctness (rtorch::autograd).
// A tiny learnable model fit with Adam; verifies the forward→graph→loss→
// backward→optimizer loop and that Adam state can be saved/restored (RTW
// checkpoint reload contract).
use rtorch::autograd::{self, Adam, Var};

fn set_pred_grad_mse(pred: &Var, target: &[f64]) {
    let mut b = pred.borrow_mut();
    for i in 0..b.data.len() {
        let t = target.get(i).copied().unwrap_or(0.0);
        b.grad[i] = 2.0 * (b.data[i] - t); // d/d_pred (pred - t)^2
    }
}

fn mse(c: &[f64], t: &[f64]) -> f64 {
    c.iter().zip(t).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / c.len() as f64
}

#[test]
fn adam_trains_a_small_model() {
    // Fit W,b so that y = X·W + b approximates target = 2*X.
    let x = autograd::from_data(vec![1.0, 2.0, 3.0, 4.0], 4, 1); // 4x1
    let target = vec![2.0, 4.0, 6.0, 8.0];
    let w = autograd::from_data(vec![0.0], 1, 1);
    let b = autograd::from_data(vec![0.0], 1, 1);

    let mut opt = Adam::new(0.05);
    let params = [w.clone(), b.clone()];

    let mut losses = Vec::new();
    for _ in 0..400 {
        let pred = autograd::add(&autograd::matmul(&x, &w), &b);
        let p = pred.borrow().data.clone();
        losses.push(mse(&p, &target));
        set_pred_grad_mse(&pred, &target);
        autograd::backward(&pred);
        opt.step(&params);
        // zero grads for the next iteration
        w.borrow_mut().grad.iter_mut().for_each(|g| *g = 0.0);
        b.borrow_mut().grad.iter_mut().for_each(|g| *g = 0.0);
    }

    assert!(losses[0] > 1.0, "initial loss too small: {}", losses[0]);
    let final_loss = losses.last().copied().unwrap();
    assert!(
        final_loss < 1e-2,
        "training did not converge: final loss {}",
        final_loss
    );
    assert!(
        final_loss < losses[0] * 0.01,
        "loss did not fall enough: {} -> {}",
        losses[0],
        final_loss
    );
    // W should approach ~2, b ~0 (2x + 0).
    assert!(
        (w.borrow().data[0] - 2.0).abs() < 0.2,
        "W={} expected ~2",
        w.borrow().data[0]
    );
    assert!(
        b.borrow().data[0].abs() < 0.2,
        "b={} expected ~0",
        b.borrow().data[0]
    );
}

#[test]
fn adam_state_roundtrip_continues() {
    let x = autograd::from_data(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
    let target = vec![2.0, 4.0, 6.0, 8.0];
    let w = autograd::from_data(vec![0.0], 1, 1);
    let b = autograd::from_data(vec![0.0], 1, 1);
    let mut opt = Adam::new(0.05);
    let params = [w.clone(), b.clone()];

    let mut step = |params: &[Var], opt: &mut Adam, w: &Var, b: &Var| {
        let pred = autograd::add(&autograd::matmul(&x, w), &b);
        set_pred_grad_mse(&pred, &target);
        autograd::backward(&pred);
        opt.step(params);
        w.borrow_mut().grad.iter_mut().for_each(|g| *g = 0.0);
        b.borrow_mut().grad.iter_mut().for_each(|g| *g = 0.0);
        mse(&pred.borrow().data, &target)
    };

    for _ in 0..200 {
        step(&params, &mut opt, &w, &b);
    }
    let before = mse(
        &{
            let pred = autograd::add(&autograd::matmul(&x, &w), &b);
            pred.borrow().data.clone()
        },
        &target,
    );

    // Save optimizer state (m/v/t) and weights, then restore into a fresh optimizer.
    let (ms, vs, t) = opt.state(&params);
    let w_snap = w.borrow().data.clone();
    let b_snap = b.borrow().data.clone();

    let w2 = autograd::from_data(w_snap, 1, 1);
    let b2 = autograd::from_data(b_snap, 1, 1);
    let params2 = [w2.clone(), b2.clone()];
    let mut opt2 = Adam::new(0.05);
    opt2.load_state(&params2, &ms, &vs, t);

    for _ in 0..20 {
        step(&params2, &mut opt2, &w2, &b2);
    }
    let after = mse(
        &{
            let pred = autograd::add(&autograd::matmul(&x, &w2), &b2);
            pred.borrow().data.clone()
        },
        &target,
    );

    assert!(
        after < before,
        "checkpoint reload did not continue improving: {} -> {}",
        before,
        after
    );
}
