// Autograd correctness (rtorch::autograd) — CPU graph backward + gradient
// accumulation across shared parameters / multiple branches.
use rtorch::autograd::{self, Var};

fn set_all_grad(v: &Var, val: f64) {
    let mut b = v.borrow_mut();
    for g in &mut b.grad {
        *g = val;
    }
}

#[test]
fn matmul_backward_grads() {
    // A[2x2] . B[2x2]=Z; loss = sum(Z) (grad of Z = all ones).
    let a = autograd::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = autograd::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let z = autograd::matmul(&a, &b);
    set_all_grad(&z, 1.0);
    autograd::backward(&z);

    // dA[i,t] = Σ_j B[t,j]; dB[t,j] = Σ_i A[i,t]
    // A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    let dA = [11.0, 15.0, 11.0, 15.0];
    let dB = [4.0, 4.0, 6.0, 6.0];
    for i in 0..4 {
        assert!(
            (a.borrow().grad[i] - dA[i]).abs() < 1e-12,
            "dA[{}]={} expected {}",
            i,
            a.borrow().grad[i],
            dA[i]
        );
        assert!(
            (b.borrow().grad[i] - dB[i]).abs() < 1e-12,
            "dB[{}]={} expected {}",
            i,
            b.borrow().grad[i],
            dB[i]
        );
    }
}

#[test]
fn tanh_backward_uses_output() {
    // y = tanh(x); dy/dx = 1 - tanh(x)^2 = 1 - y^2
    let x = autograd::from_data(vec![0.0, 0.5, 1.0, -1.5], 2, 2);
    let y = autograd::tanh(&x);
    set_all_grad(&y, 1.0);
    autograd::backward(&y);
    for i in 0..4 {
        let expected = 1.0 - y.borrow().data[i] * y.borrow().data[i];
        assert!(
            (x.borrow().grad[i] - expected).abs() < 1e-12,
            "grad[{}]={} expected {}",
            i,
            x.borrow().grad[i],
            expected
        );
    }
}

#[test]
fn shared_param_two_branches_accumulates() {
    // shared backbone p feeds two heads (two losses). Backward over the combined
    // loss must accumulate p's gradient from BOTH branches.
    let p = autograd::from_data(vec![1.0, 0.5, -0.5, 2.0], 2, 2); // shared param
    let w1 = autograd::from_data(vec![1.0, 0.0, 0.0, 1.0], 2, 2); // head A
    let w2 = autograd::from_data(vec![0.0, 1.0, 1.0, 0.0], 2, 2); // head B
    let a = autograd::matmul(&p, &w1);
    let b = autograd::matmul(&p, &w2);
    let loss = autograd::add(&a, &b); // combined loss
    set_all_grad(&loss, 1.0);
    autograd::backward(&loss);

    // d(sum(p·W))/d(p[i,k]) = Σ_j W[k,j] (shared across i); summed over both heads.
    // W1 = I => rowsum(W1)=[1,1]; W2 = [[0,1],[1,0]] => rowsum(W2)=[1,1].
    // So expected dL/d(p[i,k]) = 1 + 1 = 2 for every element.
    for i in 0..4 {
        assert!(
            (p.borrow().grad[i] - 2.0).abs() < 1e-12,
            "p.grad[{}]={} expected 2",
            i,
            p.borrow().grad[i]
        );
    }
}

#[test]
fn scale_and_add_chain() {
    // loss = 3*(x + y); d/dx = 3, d/dy = 3
    let x = autograd::from_data(vec![1.0, 2.0], 1, 2);
    let y = autograd::from_data(vec![3.0, 4.0], 1, 2);
    let s = autograd::add(&x, &y);
    let loss = autograd::scale(3.0, &s);
    set_all_grad(&loss, 1.0);
    autograd::backward(&loss);
    for i in 0..2 {
        assert!((x.borrow().grad[i] - 3.0).abs() < 1e-12);
        assert!((y.borrow().grad[i] - 3.0).abs() < 1e-12);
    }
}
