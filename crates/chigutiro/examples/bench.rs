//! `cargo run --release -p chigutiro --example bench [rows]`
//!
//! Times construction and brush moves on synthetic data shaped like a scientific table:
//! three correlated numeric columns and one categorical column.

use chigutiro::{Axis, Binning, Crossfilter, Keys};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut unit = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    let gauss = |u: &mut dyn FnMut() -> f64| {
        (-2.0 * u().max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u()).cos()
    };
    let x: Vec<f64> = (0..n).map(|_| gauss(&mut unit)).collect();
    let y: Vec<f64> = x.iter().map(|&x| 0.6 * x + 0.8 * gauss(&mut unit)).collect();
    let z: Vec<f64> = x.iter().zip(&y).map(|(&x, &y)| x * y + 0.3 * gauss(&mut unit)).collect();
    let k: Vec<u32> = (0..n).map(|_| (unit() * 12.0) as u32).collect();

    let t = Instant::now();
    let mut cf = Crossfilter::new(n).unwrap();
    let dx = cf.add_dimension(Keys::Numeric(x)).unwrap();
    let dy = cf.add_dimension(Keys::Numeric(y)).unwrap();
    let dz = cf.add_dimension(Keys::Numeric(z)).unwrap();
    let dk = cf.add_dimension(Keys::Categorical { codes: k, cardinality: 12 }).unwrap();
    let build = t.elapsed();

    let t = Instant::now();
    let auto = |dim, bins| Axis { dim, min: f64::NAN, width: f64::NAN, bins };
    cf.add_group(Binning::Histogram { dim: dx, min: f64::NAN, width: f64::NAN, bins: 60 }).unwrap();
    cf.add_group(Binning::Histogram { dim: dz, min: f64::NAN, width: f64::NAN, bins: 60 }).unwrap();
    cf.add_group(Binning::Category { dim: dk, cardinality: 0 }).unwrap();
    cf.add_group(Binning::Grid { x: auto(dx, 256), y: auto(dy, 256) }).unwrap();
    let groups = t.elapsed();

    println!("rows {n}, 4 dimensions, 4 groups (2 histograms, 1 category, 1 256x256 grid)");
    println!("build dimensions   {:>9.1} ms", build.as_secs_f64() * 1e3);
    println!("build groups       {:>9.1} ms", groups.as_secs_f64() * 1e3);

    // A brush dragged in small steps: the interactive case.
    let mut times = Vec::new();
    let mut crossed = 0;
    for i in 0..200 {
        let lo = -1.0 + i as f64 * 0.005;
        let t = Instant::now();
        crossed += cf.filter_range(dx, lo, lo + 1.0).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    report("drag x-brush (200 steps)", &mut times, crossed / 200, n);

    // With other filters active, so the group updates actually fire.
    cf.filter_range(dy, -0.5, 1.5).unwrap();
    cf.filter_codes(dk, &[0, 2, 4, 6, 8, 10]).unwrap();
    let mut times = Vec::new();
    let mut crossed = 0;
    for i in 0..200 {
        let lo = -1.0 + i as f64 * 0.005;
        let t = Instant::now();
        crossed += cf.filter_range(dz, lo, lo + 1.0).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    report("drag z-brush, 3 filters on", &mut times, crossed / 200, n);

    let mut times = Vec::new();
    let mut crossed = 0;
    for _ in 0..10 {
        let t = Instant::now();
        crossed += cf.clear_filter(dz).unwrap();
        times.push(t.elapsed().as_secs_f64() * 1e3);
        cf.filter_range(dz, -0.2, 0.2).unwrap();
    }
    report("clear a narrow filter (worst case)", &mut times, crossed / 10, n);
    println!("selected now       {:>9}", cf.selected_count());
}

fn report(label: &str, times: &mut [f64], delta: usize, n: usize) {
    times.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
    println!(
        "{label:<34} median {:>7.3} ms  p95 {:>7.3} ms  max {:>7.3} ms  (rows crossed per step {delta}, {:.2}% of n)",
        pct(0.5),
        pct(0.95),
        pct(1.0),
        100.0 * delta as f64 / n as f64
    );
}
