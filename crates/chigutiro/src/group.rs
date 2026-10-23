//! Groups: counts per bin over the rows that pass every filter except the group's own.

use crate::dimension::{Dimension, Keys};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Binning {
    /// Equal-width bins over `[min, min + width * bins]` of a numeric dimension.
    Histogram { dim: usize, min: f64, width: f64, bins: u32 },
    /// One bin per code of a categorical dimension.
    Category { dim: usize, cardinality: u32 },
    /// A `nx * ny` grid over two numeric dimensions, row-major in y.
    Grid { x: Axis, y: Axis },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis {
    pub dim: usize,
    pub min: f64,
    pub width: f64,
    pub bins: u32,
}

#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) binning: Binning,
    /// Dimensions whose filters this group ignores.
    pub(crate) own: u32,
    pub(crate) counts: Vec<u32>,
}

fn equal_width(v: f64, min: f64, width: f64, bins: u32) -> Option<usize> {
    if v.is_nan() || v < min || v > min + width * bins as f64 {
        return None;
    }
    Some((((v - min) / width) as usize).min(bins as usize - 1))
}

fn numeric(dims: &[Dimension], dim: usize, row: usize) -> f64 {
    match &dims[dim].keys {
        Keys::Numeric(values) => values[row],
        Keys::Categorical { .. } => unreachable!("binning validated at construction"),
    }
}

impl Group {
    pub(crate) fn new(binning: Binning) -> Self {
        let (own, len) = match binning {
            Binning::Histogram { dim, bins, .. } => (1 << dim, bins as usize),
            Binning::Category { dim, cardinality } => (1 << dim, cardinality as usize),
            Binning::Grid { x, y } => ((1 << x.dim) | (1 << y.dim), (x.bins * y.bins) as usize),
        };
        Group { binning, own, counts: vec![0; len] }
    }

    #[inline]
    pub(crate) fn bin(&self, dims: &[Dimension], row: usize) -> Option<usize> {
        match self.binning {
            Binning::Histogram { dim, min, width, bins } => {
                equal_width(numeric(dims, dim, row), min, width, bins)
            }
            Binning::Category { dim, .. } => match &dims[dim].keys {
                Keys::Categorical { codes, .. } => Some(codes[row] as usize),
                Keys::Numeric(_) => unreachable!("binning validated at construction"),
            },
            Binning::Grid { x, y } => {
                let bx = equal_width(numeric(dims, x.dim, row), x.min, x.width, x.bins)?;
                let by = equal_width(numeric(dims, y.dim, row), y.min, y.width, y.bins)?;
                Some(by * x.bins as usize + bx)
            }
        }
    }

    /// Recounts from scratch: one sequential pass, specialised per binning.
    pub(crate) fn recount(&mut self, dims: &[Dimension], masks: &[u32]) {
        let ignore = !self.own;
        let counts = &mut self.counts;
        counts.fill(0);
        let values = |dim: usize| match &dims[dim].keys {
            Keys::Numeric(v) => v.as_slice(),
            Keys::Categorical { .. } => unreachable!("binning validated at construction"),
        };
        match self.binning {
            Binning::Histogram { dim, min, width, bins } => {
                for (&m, &v) in masks.iter().zip(values(dim)) {
                    if m & ignore == 0 {
                        if let Some(b) = equal_width(v, min, width, bins) {
                            counts[b] += 1;
                        }
                    }
                }
            }
            Binning::Category { dim, .. } => {
                let Keys::Categorical { codes, .. } = &dims[dim].keys else {
                    unreachable!("binning validated at construction")
                };
                for (&m, &c) in masks.iter().zip(codes) {
                    if m & ignore == 0 {
                        counts[c as usize] += 1;
                    }
                }
            }
            Binning::Grid { x, y } => {
                for ((&m, &vx), &vy) in masks.iter().zip(values(x.dim)).zip(values(y.dim)) {
                    if m & ignore == 0 {
                        if let (Some(bx), Some(by)) =
                            (equal_width(vx, x.min, x.width, x.bins), equal_width(vy, y.min, y.width, y.bins))
                        {
                            counts[by * x.bins as usize + bx] += 1;
                        }
                    }
                }
            }
        }
    }
}
