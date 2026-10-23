//! # chigutiro
//!
//! Incremental multidimensional crossfiltering over columnar data.
//!
//! Every dimension keeps its rows in key order, and its filter admits a set of positions
//! in that order. Every row carries a bitmask of the dimensions that currently reject it.
//! Changing one filter computes the positions that entered or left its admitted set and
//! touches only those rows. That is what makes a brush move cost O(Δ), not O(n):
//!
//! * the row's bit for that dimension is flipped;
//! * each group that does not ignore that dimension adds or removes the row's bin;
//! * the row's overall selected state is updated, and recorded in [`Crossfilter::changed`]
//!   when it flips.
//!
//! A group counts the rows that pass every filter *except* those on its own dimensions.
//! So a histogram keeps showing its full distribution while its own brush narrows the
//! others, as in crossfilter.js.
//!
//! Limits: at most 32 dimensions, and at most `u32::MAX` rows.

mod dimension;
mod group;
mod span;

pub use dimension::Keys;
pub use group::{Axis, Binning};
pub use span::Span;

use dimension::{Dimension, Predicate};
use group::Group;

pub const MAX_DIMENSIONS: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    LengthMismatch { expected: usize, got: usize },
    TooManyDimensions,
    TooManyRows,
    NoSuchDimension(usize),
    NoSuchGroup(usize),
    WrongKind { expected: &'static str },
    InvalidRange { lo: f64, hi: f64 },
    CodeOutOfRange { code: u32, cardinality: u32 },
    InvalidBinning(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::LengthMismatch { expected, got } => {
                write!(f, "column has {got} rows, the crossfilter has {expected}")
            }
            Error::TooManyDimensions => write!(f, "at most {MAX_DIMENSIONS} dimensions"),
            Error::TooManyRows => write!(f, "at most {} rows", u32::MAX),
            Error::NoSuchDimension(d) => write!(f, "no dimension {d}"),
            Error::NoSuchGroup(g) => write!(f, "no group {g}"),
            Error::WrongKind { expected } => write!(f, "dimension is not {expected}"),
            Error::InvalidRange { lo, hi } => write!(f, "invalid range [{lo}, {hi}]"),
            Error::CodeOutOfRange { code, cardinality } => {
                write!(f, "code {code} is outside 0..{cardinality}")
            }
            Error::InvalidBinning(why) => write!(f, "invalid binning: {why}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub struct Crossfilter {
    rows: usize,
    dims: Vec<Dimension>,
    groups: Vec<Group>,
    /// Per row: bit `d` set means dimension `d`'s filter rejects the row.
    masks: Vec<u32>,
    /// Per row: 1 when the row passes every filter.
    selected: Vec<u8>,
    selected_count: usize,
    /// Rows whose selected state flipped during the last filter change.
    changed: Vec<u32>,
    entering: Vec<Span>,
    leaving: Vec<Span>,
    touched: Vec<u32>,
    /// Above this many crossing rows, a filter change is applied by a sequential rescan.
    scan_threshold: usize,
}

impl Crossfilter {
    pub fn new(rows: usize) -> Result<Self, Error> {
        if rows > u32::MAX as usize {
            return Err(Error::TooManyRows);
        }
        Ok(Crossfilter {
            rows,
            dims: Vec::new(),
            groups: Vec::new(),
            masks: vec![0; rows],
            selected: vec![1; rows],
            selected_count: rows,
            changed: Vec::new(),
            entering: Vec::new(),
            leaving: Vec::new(),
            touched: Vec::new(),
            scan_threshold: rows / 8,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Sets how many crossing rows make a filter change switch from visiting those rows
    /// to rescanning the table. The default is an eighth of the rows.
    pub fn set_scan_threshold(&mut self, rows: usize) {
        self.scan_threshold = rows;
    }

    /// Adds an unfiltered dimension and returns its index.
    pub fn add_dimension(&mut self, keys: Keys) -> Result<usize, Error> {
        if keys.len() != self.rows {
            return Err(Error::LengthMismatch { expected: self.rows, got: keys.len() });
        }
        if self.dims.len() == MAX_DIMENSIONS {
            return Err(Error::TooManyDimensions);
        }
        self.dims.push(Dimension::new(keys)?);
        Ok(self.dims.len() - 1)
    }

    /// Smallest and largest finite value of a numeric dimension.
    pub fn extent(&self, dim: usize) -> Result<Option<(f64, f64)>, Error> {
        let d = self.dim(dim)?;
        match d.keys {
            Keys::Numeric(_) => Ok(d.extent),
            Keys::Categorical { .. } => Err(Error::WrongKind { expected: "numeric" }),
        }
    }

    /// Admits rows whose value lies in `[lo, hi]`. Missing (NaN) values are rejected.
    pub fn filter_range(&mut self, dim: usize, lo: f64, hi: f64) -> Result<usize, Error> {
        let selection = self.dim(dim)?.range_selection(lo, hi)?;
        Ok(self.apply(dim, selection))
    }

    /// Admits rows whose category code is in `codes`.
    pub fn filter_codes(&mut self, dim: usize, codes: &[u32]) -> Result<usize, Error> {
        let selection = self.dim(dim)?.set_selection(codes)?;
        Ok(self.apply(dim, selection))
    }

    /// Removes the filter on `dim`, admitting every row including missing values.
    pub fn clear_filter(&mut self, dim: usize) -> Result<usize, Error> {
        let selection = self.dim(dim)?.full_selection();
        Ok(self.apply(dim, selection))
    }

    /// Adds a group and returns its index. Histogram and grid domains may be `NaN` to
    /// mean "the dimension's extent".
    pub fn add_group(&mut self, binning: Binning) -> Result<usize, Error> {
        let binning = self.resolve(binning)?;
        let mut group = Group::new(binning);
        group.recount(&self.dims, &self.masks);
        self.groups.push(group);
        Ok(self.groups.len() - 1)
    }

    pub fn group_counts(&self, group: usize) -> Result<&[u32], Error> {
        self.groups.get(group).map(|g| g.counts.as_slice()).ok_or(Error::NoSuchGroup(group))
    }

    /// The binning a group was created with, with any default domain filled in.
    pub fn group_binning(&self, group: usize) -> Result<Binning, Error> {
        self.groups.get(group).map(|g| g.binning).ok_or(Error::NoSuchGroup(group))
    }

    /// Per row: 1 when the row passes every filter, else 0.
    pub fn selected(&self) -> &[u8] {
        &self.selected
    }

    pub fn selected_count(&self) -> usize {
        self.selected_count
    }

    /// Rows whose selected state flipped during the last filter change.
    pub fn changed(&self) -> &[u32] {
        &self.changed
    }

    fn dim(&self, dim: usize) -> Result<&Dimension, Error> {
        self.dims.get(dim).ok_or(Error::NoSuchDimension(dim))
    }

    fn resolve(&self, binning: Binning) -> Result<Binning, Error> {
        let axis = |dim: usize, min: f64, width: f64, bins: u32| -> Result<Axis, Error> {
            if bins == 0 {
                return Err(Error::InvalidBinning("bins must be positive"));
            }
            let (min, width) = if min.is_nan() || width.is_nan() {
                let (lo, hi) = self.extent(dim)?.unwrap_or((0.0, 1.0));
                let span = if hi > lo { hi - lo } else { 1.0 };
                (lo, span / bins as f64)
            } else {
                self.extent(dim)?;
                (min, width)
            };
            if !(width > 0.0) || !min.is_finite() || !width.is_finite() {
                return Err(Error::InvalidBinning("width must be positive and finite"));
            }
            Ok(Axis { dim, min, width, bins })
        };
        match binning {
            Binning::Histogram { dim, min, width, bins } => {
                let a = axis(dim, min, width, bins)?;
                Ok(Binning::Histogram { dim, min: a.min, width: a.width, bins })
            }
            Binning::Category { dim, .. } => match self.dim(dim)?.keys {
                Keys::Categorical { cardinality, .. } => Ok(Binning::Category { dim, cardinality }),
                Keys::Numeric(_) => Err(Error::WrongKind { expected: "categorical" }),
            },
            Binning::Grid { x, y } => {
                if x.dim == y.dim {
                    return Err(Error::InvalidBinning("grid needs two different dimensions"));
                }
                Ok(Binning::Grid {
                    x: axis(x.dim, x.min, x.width, x.bins)?,
                    y: axis(y.dim, y.min, y.width, y.bins)?,
                })
            }
        }
    }

    /// Replaces `dim`'s filter and propagates the change. Returns how many rows crossed
    /// the filter boundary.
    fn apply(&mut self, dim: usize, (selection, predicate): (Vec<Span>, Predicate)) -> usize {
        span::difference(&selection, &self.dims[dim].selection, &mut self.entering);
        span::difference(&self.dims[dim].selection, &selection, &mut self.leaving);
        let crossed: usize = self.entering.iter().chain(&self.leaving).map(|&(s, e)| (e - s) as usize).sum();
        self.dims[dim].selection = selection;
        self.dims[dim].predicate = predicate;
        self.changed.clear();
        if crossed > self.scan_threshold {
            self.rescan(dim);
        } else {
            self.update(dim);
        }
        crossed
    }

    /// Small changes: visit only the rows that crossed, in row order so memory is walked
    /// forwards rather than at random.
    fn update(&mut self, dim: usize) {
        let bit = 1u32 << dim;
        let order = &self.dims[dim].order;
        for (spans, reject) in [(&self.entering, false), (&self.leaving, true)] {
            self.touched.clear();
            for &(start, end) in spans {
                self.touched.extend_from_slice(&order[start as usize..end as usize]);
            }
            self.touched.sort_unstable();
            for &row in &self.touched {
                let row = row as usize;
                let old = self.masks[row];
                let new = if reject { old | bit } else { old & !bit };
                self.masks[row] = new;
                for g in &mut self.groups {
                    if g.own & bit != 0 {
                        continue;
                    }
                    let was = old & !g.own == 0;
                    if was != (new & !g.own == 0) {
                        if let Some(b) = g.bin(&self.dims, row) {
                            if was {
                                g.counts[b] -= 1;
                            } else {
                                g.counts[b] += 1;
                            }
                        }
                    }
                }
                if (old == 0) != (new == 0) {
                    let now = (new == 0) as u8;
                    self.selected[row] = now;
                    if now == 1 {
                        self.selected_count += 1;
                    } else {
                        self.selected_count -= 1;
                    }
                    self.changed.push(row as u32);
                }
            }
        }
    }

    /// Large changes: one sequential pass resets the dimension's bit from its predicate,
    /// then one sequential pass per affected group recounts it. Each loop is specialised
    /// to its predicate or binning so the compiler can vectorise it.
    fn rescan(&mut self, dim: usize) {
        let bit = 1u32 << dim;
        let masks = &mut self.masks;
        match (&self.dims[dim].predicate, &self.dims[dim].keys) {
            (Predicate::All, _) => masks.iter_mut().for_each(|m| *m &= !bit),
            (Predicate::Range(lo, hi), Keys::Numeric(v)) => {
                let (lo, hi) = (*lo, *hi);
                for (m, &x) in masks.iter_mut().zip(v) {
                    *m = if lo <= x && x <= hi { *m & !bit } else { *m | bit };
                }
            }
            (Predicate::Codes(admitted), Keys::Categorical { codes, .. }) => {
                for (m, &c) in masks.iter_mut().zip(codes) {
                    *m = if admitted[c as usize] { *m & !bit } else { *m | bit };
                }
            }
            _ => unreachable!("predicate kind matches keys"),
        }

        let mut count = 0;
        for (row, (s, &m)) in self.selected.iter_mut().zip(masks.iter()).enumerate() {
            let now = (m == 0) as u8;
            if *s != now {
                *s = now;
                self.changed.push(row as u32);
            }
            count += now as usize;
        }
        self.selected_count = count;

        for g in self.groups.iter_mut().filter(|g| g.own & bit == 0) {
            g.recount(&self.dims, &self.masks);
        }
    }
}

#[cfg(test)]
mod tests;
