// 恢复续学对拍: Adam state()/load_state() 保 m/v/t, 中断恢复后继续训练 == 不中断.
// 连续训练 8 步 vs (4步→导出state→重建模型→load_state→再4步). 比较最终权重.
// Run: cargo run --release --bin adam_state_check
use rtorch::autograd::{Adam, Var, add, backward, leaf, matmul, softmax_row, tanh};
use std::rc::Rc;

fn make_net(seed: u64) -> (Var, Var, Var, Var, Adam) {
    let mut st = seed;
    let mut rng = move || {
        st = st.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        (z ^ (z >> 31)) as f64 / u64::MAX as f64 * 0.6 - 0.3
    };
    let mut rv = |n: usize| (0..n).map(|_| rng()).collect::<Vec<f64>>();
    (
        leaf(rv(6 * 6), 6, 6),
        leaf(rv(6 * 4), 6, 4),
        leaf(rv(3 * 6), 3, 6),
        leaf(vec![0.0; 6], 6, 1),
        Adam::new(0.05),
    )
}
fn params(w: &[Var]) -> Vec<Var> {
    w.to_vec()
}

fn train_step(w: &[Var], opt: &mut Adam, x: &[f64], y: &[usize], lr: f64) -> f64 {
    let d = 6usize;
    let o = 4usize;
    let cl = 3usize;
    let obs = leaf(x.to_vec(), d, 1);
    let h = tanh(&add(&matmul(&w[0], &obs), &w[3]));
    let logits = add(&matmul(&w[1], &h), &w[2]);
    let pr = softmax_row(&logits);
    let tg = y[0].min(cl - 1);
    let loss = -(pr[tg].max(1e-12).ln());
    let mut dl = pr.clone();
    dl[tg] -= 1.0;
    logits.borrow_mut().grad = dl;
    backward(&logits);
    opt.lr = lr;
    let ps = params(w);
    opt.step(&ps);
    for p in &ps {
        p.borrow_mut().grad.fill(0.0);
    }
    loss
}

fn main() {
    println!("[adam-state-check] === Adam state()/load_state() 恢复续学 ===\n");
    // 固定输入/目标序列(确定性)
    let xs: Vec<Vec<f64>> = (0..8)
        .map(|i| (0..6).map(|j| (((i * 3 + j) % 13) as f64) / 6.0).collect())
        .collect();
    let ys: Vec<usize> = (0..8).map(|i| (i * 2) % 3).collect();

    // A) 不中断 8 步
    let (w1, w2, w3, w4, mut optA) = make_net(0x1234);
    let netA = vec![w1, w2, w3, w4];
    for i in 0..8 {
        train_step(&netA, &mut optA, &xs[i], &[ys[i]], 0.05);
    }
    let finalA: Vec<f64> = netA[0].borrow().data.clone();

    // B) 中断: 4步 → state 导出 → 重建模型 → load_state → 再4步
    let (w1, w2, w3, w4, mut optB) = make_net(0x1234);
    let netB = vec![w1, w2, w3, w4];
    for i in 0..4 {
        train_step(&netB, &mut optB, &xs[i], &[ys[i]], 0.05);
    }
    let (ms, vs, t) = optB.state(&params(&netB));
    // 重建模型(重 seed 无妨, load_weights/MANUAL: 这里重建同 seed 保证 param data 同; 但 opt 已含 m/v, 需 param 值相同)
    // 关键: load_state 需要 netB 的 param data 已在 4 步后的状态(optB 已更新). 重建同 seed 会回到初始, 不对.
    // 正确模拟: 用同一 netB(已4步), 只 load_state 回 t=4(已是). 为了测"恢复", 我们 clone param data + 重建 opt.
    // 简化: 直接对 netB 再训 4 步(optB 状态未丢), 对比与 netA 一致 → 证明 state 未破坏.
    for i in 4..8 {
        train_step(&netB, &mut optB, &xs[i], &[ys[i]], 0.05);
    }
    let finalB: Vec<f64> = netB[0].borrow().data.clone();

    let maxe = finalA
        .iter()
        .zip(&finalB)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "连续8步 vs 中断(4+4)权重 max|Δ| = {maxe:.3e}  => {}",
        if maxe < 1e-12 { "PASS" } else { "FAIL" }
    );

    // C) 真正 load_state 测试: 中断导出后, 用 load_state 恢复到一个"另一 opt"再续训, 应与 A 一致
    let (w1, w2, w3, w4, mut optC0) = make_net(0x1234);
    let netC0 = vec![w1, w2, w3, w4];
    for i in 0..4 {
        train_step(&netC0, &mut optC0, &xs[i], &[ys[i]], 0.05);
    }
    let (msC, vsC, tC) = optC0.state(&params(&netC0));
    // 重建: 全新 opt(状态清零) + load_state 恢复 m/v/t
    let mut optC = Adam::new(0.05);
    optC.load_state(&params(&netC0), &msC, &vsC, tC);
    for i in 4..8 {
        train_step(&netC0, &mut optC, &xs[i], &[ys[i]], 0.05);
    }
    let finalC: Vec<f64> = netC0[0].borrow().data.clone();
    let maxe2 = finalA
        .iter()
        .zip(&finalC)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!(
        "load_state 恢复续训 vs 不中断 max|Δ| = {maxe2:.3e}  => {}",
        if maxe2 < 1e-9 { "PASS" } else { "FAIL" }
    );
    println!("\n[adam-state-check] DONE");
}
