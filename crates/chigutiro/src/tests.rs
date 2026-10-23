use super::*;

/// xorshift64*: deterministic, so a failure reproduces from its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Debug)]
enum Filter {
    None,
    Range(f64, f64),
    Codes(Vec<u32>),
}

/// The brute-force model: recompute everything from the filters.
struct Model {
    keys: Vec<Keys>,
    filters: Vec<Filter>,
}

impl Model {
    fn passes(&self, dim: usize, row: usize) -> bool {
        match (&self.filters[dim], &self.keys[dim]) {
            (Filter::None, _) => true,
            (Filter::Range(lo, hi), Keys::Numeric(v)) => *lo <= v[row] && v[row] <= *hi,
            (Filter::Codes(cs), Keys::Categorical { codes, .. }) => cs.contains(&codes[row]),
            _ => unreachable!(),
        }
    }

    fn passes_except(&self, row: usize, own: u32) -> bool {
        (0..self.keys.len()).all(|d| own & (1 << d) != 0 || self.passes(d, row))
    }
}

fn numeric_column(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| match rng.below(20) {
            0 => f64::NAN,
            1 => 0.5, // repeated values exercise ties at range edges
            _ => (rng.unit() * 10.0).floor() / 10.0,
        })
        .collect()
}

fn check(cf: &Crossfilter, model: &Model, groups: &[usize]) {
    let n = cf.rows();
    let want: Vec<u8> = (0..n).map(|r| model.passes_except(r, 0) as u8).collect();
    assert_eq!(cf.selected(), want.as_slice(), "selected mask");
    assert_eq!(cf.selected_count(), want.iter().filter(|&&s| s == 1).count());
    for &g in groups {
        let group = &cf.groups[g];
        let mut counts = vec![0u32; group.counts.len()];
        for row in 0..n {
            if model.passes_except(row, group.own) {
                if let Some(b) = group.bin(&cf.dims, row) {
                    counts[b] += 1;
                }
            }
        }
        assert_eq!(cf.group_counts(g).unwrap(), counts.as_slice(), "group {g}");
    }
}

#[test]
fn incremental_updates_match_brute_force() {
    // 0 forces every change through the rescan, MAX through the row-by-row update,
    // and None keeps the default, which mixes both.
    for threshold in [Some(0), Some(usize::MAX), None] {
        brute_force(threshold);
    }
}

fn brute_force(threshold: Option<usize>) {
    for seed in 1..=40u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let n = 1 + rng.below(400) as usize;
        let mut cf = Crossfilter::new(n).unwrap();
        if let Some(t) = threshold {
            cf.set_scan_threshold(t);
        }
        let mut model = Model { keys: Vec::new(), filters: Vec::new() };

        for d in 0..4 {
            let keys = if d % 2 == 0 {
                Keys::Numeric(numeric_column(&mut rng, n))
            } else {
                let cardinality = 1 + rng.below(6) as u32;
                let codes = (0..n).map(|_| rng.below(cardinality as u64) as u32).collect();
                Keys::Categorical { codes, cardinality }
            };
            cf.add_dimension(keys.clone()).unwrap();
            model.keys.push(keys);
            model.filters.push(Filter::None);
        }

        let mut groups = vec![
            cf.add_group(Binning::Histogram { dim: 0, min: f64::NAN, width: f64::NAN, bins: 7 }).unwrap(),
            cf.add_group(Binning::Category { dim: 1, cardinality: 0 }).unwrap(),
            cf.add_group(Binning::Histogram { dim: 2, min: 0.2, width: 0.15, bins: 4 }).unwrap(),
        ];
        let grid = Axis { dim: 0, min: f64::NAN, width: f64::NAN, bins: 3 };
        groups.push(cf.add_group(Binning::Grid { x: grid, y: Axis { dim: 2, ..grid } }).unwrap());
        check(&cf, &model, &groups);

        for step in 0..60 {
            let dim = rng.below(4) as usize;
            let filter = match (rng.below(4), &model.keys[dim]) {
                (0, _) => Filter::None,
                (_, Keys::Numeric(_)) => {
                    let a = (rng.unit() * 11.0 - 0.5).floor() / 10.0;
                    let b = (rng.unit() * 11.0 - 0.5).floor() / 10.0;
                    Filter::Range(a.min(b), a.max(b))
                }
                (_, Keys::Categorical { cardinality, .. }) => Filter::Codes(
                    (0..*cardinality).filter(|_| rng.below(2) == 0).collect(),
                ),
            };
            let before = cf.selected().to_vec();
            match &filter {
                Filter::None => cf.clear_filter(dim),
                Filter::Range(lo, hi) => cf.filter_range(dim, *lo, *hi),
                Filter::Codes(cs) => cf.filter_codes(dim, cs),
            }
            .unwrap();
            model.filters[dim] = filter;
            check(&cf, &model, &groups);

            let mut flipped: Vec<u32> = (0..n as u32)
                .filter(|&r| before[r as usize] != cf.selected()[r as usize])
                .collect();
            let mut reported = cf.changed().to_vec();
            reported.sort_unstable();
            flipped.sort_unstable();
            assert_eq!(reported, flipped, "seed {seed} step {step}: changed rows");

            if step == 30 {
                // A group added mid-session must start from the current filters.
                groups.push(cf.add_group(Binning::Category { dim: 3, cardinality: 0 }).unwrap());
                check(&cf, &model, &groups);
            }
        }
    }
}

#[test]
fn missing_values_are_rejected_by_ranges_and_restored_on_clear() {
    let mut cf = Crossfilter::new(4).unwrap();
    let d = cf.add_dimension(Keys::Numeric(vec![1.0, f64::NAN, 3.0, -0.0])).unwrap();
    assert_eq!(cf.extent(d).unwrap(), Some((-0.0, 3.0)));
    cf.filter_range(d, 0.0, f64::INFINITY).unwrap();
    assert_eq!(cf.selected(), &[1, 0, 1, 1]); // -0.0 lies in [0, inf]
    cf.clear_filter(d).unwrap();
    assert_eq!(cf.selected_count(), 4);
}

#[test]
fn a_brush_move_touches_only_rows_that_cross_its_edges() {
    let n = 1000;
    let mut cf = Crossfilter::new(n).unwrap();
    let d = cf.add_dimension(Keys::Numeric((0..n).map(|i| i as f64).collect())).unwrap();
    cf.filter_range(d, 100.0, 199.0).unwrap();
    assert_eq!(cf.filter_range(d, 110.0, 209.0).unwrap(), 20);
    assert_eq!(cf.changed().len(), 20);
    assert_eq!(cf.selected_count(), 100);
}

#[test]
fn errors_are_reported_not_panicked() {
    let mut cf = Crossfilter::new(3).unwrap();
    assert_eq!(cf.add_dimension(Keys::Numeric(vec![1.0])), Err(Error::LengthMismatch { expected: 3, got: 1 }));
    let num = cf.add_dimension(Keys::Numeric(vec![1.0, 2.0, 3.0])).unwrap();
    let cat = cf.add_dimension(Keys::Categorical { codes: vec![0, 1, 1], cardinality: 2 }).unwrap();
    assert_eq!(cf.filter_range(num, 2.0, 1.0), Err(Error::InvalidRange { lo: 2.0, hi: 1.0 }));
    assert_eq!(cf.filter_range(cat, 0.0, 1.0), Err(Error::WrongKind { expected: "numeric" }));
    assert_eq!(cf.filter_codes(cat, &[5]), Err(Error::CodeOutOfRange { code: 5, cardinality: 2 }));
    assert!(cf.add_group(Binning::Category { dim: num, cardinality: 0 }).is_err());
    assert!(cf.add_group(Binning::Histogram { dim: num, min: 0.0, width: 0.0, bins: 3 }).is_err());
    assert_eq!(cf.clear_filter(9), Err(Error::NoSuchDimension(9)));
}
