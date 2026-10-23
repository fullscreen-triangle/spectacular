//! A dimension: one column, its rows in key order, and the positions its filter admits.

use crate::span::{push_merged, Span};
use crate::Error;

/// The values a dimension filters on.
#[derive(Debug, Clone)]
pub enum Keys {
    /// Real values. NaN means "missing": it sorts last and no range filter admits it.
    Numeric(Vec<f64>),
    /// Category codes in `0..cardinality`.
    Categorical { codes: Vec<u32>, cardinality: u32 },
}

impl Keys {
    pub fn len(&self) -> usize {
        match self {
            Keys::Numeric(v) => v.len(),
            Keys::Categorical { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What a dimension's filter admits, kept so a large change can be applied by a scan.
#[derive(Debug, Clone)]
pub(crate) enum Predicate {
    All,
    Range(f64, f64),
    /// Indexed by code.
    Codes(Vec<bool>),
}

#[derive(Debug)]
pub(crate) struct Dimension {
    pub(crate) keys: Keys,
    /// Row ids sorted by key.
    pub(crate) order: Vec<u32>,
    /// Categorical: position where each code starts, `cardinality + 1` entries.
    offsets: Vec<u32>,
    /// Numeric: positions at or past this hold NaN.
    finite_end: u32,
    /// Positions the current filter admits.
    pub(crate) selection: Vec<Span>,
    pub(crate) predicate: Predicate,
    /// Numeric: smallest and largest finite value, if any.
    pub(crate) extent: Option<(f64, f64)>,
}

impl Dimension {
    pub(crate) fn new(keys: Keys) -> Result<Self, Error> {
        let n = keys.len();
        let all = if n > 0 { vec![(0, n as u32)] } else { Vec::new() };
        let mut dim = match &keys {
            Keys::Numeric(values) => {
                let mut pairs: Vec<(f64, u32)> = values
                    .iter()
                    .enumerate()
                    .map(|(row, &v)| (if v.is_nan() { f64::NAN } else { v }, row as u32))
                    .collect();
                // total_cmp puts positive NaN after +inf, which is where missing values belong.
                pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
                let finite_end = pairs.partition_point(|p| !p.0.is_nan()) as u32;
                let extent = (finite_end > 0).then(|| (pairs[0].0, pairs[finite_end as usize - 1].0));
                Dimension {
                    order: pairs.into_iter().map(|p| p.1).collect(),
                    offsets: Vec::new(),
                    finite_end,
                    selection: Vec::new(),
                    predicate: Predicate::All,
                    extent,
                    keys: Keys::Numeric(Vec::new()),
                }
            }
            Keys::Categorical { codes, cardinality } => {
                let card = *cardinality as usize;
                let mut offsets = vec![0u32; card + 1];
                for &c in codes {
                    if c >= *cardinality {
                        return Err(Error::CodeOutOfRange { code: c, cardinality: *cardinality });
                    }
                    offsets[c as usize + 1] += 1;
                }
                for i in 0..card {
                    offsets[i + 1] += offsets[i];
                }
                let mut next = offsets.clone();
                let mut order = vec![0u32; n];
                for (row, &c) in codes.iter().enumerate() {
                    order[next[c as usize] as usize] = row as u32;
                    next[c as usize] += 1;
                }
                Dimension {
                    order,
                    offsets,
                    finite_end: n as u32,
                    selection: Vec::new(),
                    predicate: Predicate::All,
                    extent: None,
                    keys: Keys::Categorical { codes: Vec::new(), cardinality: *cardinality },
                }
            }
        };
        dim.keys = keys;
        dim.selection = all;
        Ok(dim)
    }

    /// Positions holding values in `[lo, hi]`.
    pub(crate) fn range_selection(&self, lo: f64, hi: f64) -> Result<(Vec<Span>, Predicate), Error> {
        let Keys::Numeric(values) = &self.keys else {
            return Err(Error::WrongKind { expected: "numeric" });
        };
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Err(Error::InvalidRange { lo, hi });
        }
        let finite = &self.order[..self.finite_end as usize];
        let start = finite.partition_point(|&r| values[r as usize] < lo) as u32;
        let end = finite.partition_point(|&r| values[r as usize] <= hi) as u32;
        let spans = if start < end { vec![(start, end)] } else { Vec::new() };
        Ok((spans, Predicate::Range(lo, hi)))
    }

    /// Positions holding any of `codes`.
    pub(crate) fn set_selection(&self, codes: &[u32]) -> Result<(Vec<Span>, Predicate), Error> {
        let Keys::Categorical { cardinality, .. } = &self.keys else {
            return Err(Error::WrongKind { expected: "categorical" });
        };
        let mut wanted = codes.to_vec();
        wanted.sort_unstable();
        wanted.dedup();
        let mut spans = Vec::with_capacity(wanted.len());
        let mut admitted = vec![false; *cardinality as usize];
        for c in wanted {
            if c >= *cardinality {
                return Err(Error::CodeOutOfRange { code: c, cardinality: *cardinality });
            }
            admitted[c as usize] = true;
            push_merged(&mut spans, (self.offsets[c as usize], self.offsets[c as usize + 1]));
        }
        Ok((spans, Predicate::Codes(admitted)))
    }

    pub(crate) fn full_selection(&self) -> (Vec<Span>, Predicate) {
        let n = self.order.len() as u32;
        (if n > 0 { vec![(0, n)] } else { Vec::new() }, Predicate::All)
    }
}
