//! WebAssembly bindings for chigutiro.
//!
//! Small results (group counts, extents) are returned as copies. The per-row arrays
//! (`selected`, `changed`) are exposed as pointers into WASM memory, so JS can view them
//! without copying: `new Uint8Array(wasmMemory().buffer, cf.selectedPtr(), cf.rows())`. A view
//! is valid until the next call that may allocate, so take a fresh one after each filter
//! change.

use chigutiro::{Axis, Binning, Keys};
use wasm_bindgen::prelude::*;

fn js(e: chigutiro::Error) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
pub struct Crossfilter {
    inner: chigutiro::Crossfilter,
}

#[wasm_bindgen]
impl Crossfilter {
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize) -> Result<Crossfilter, JsError> {
        Ok(Crossfilter { inner: chigutiro::Crossfilter::new(rows).map_err(js)? })
    }

    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Adds a numeric dimension. NaN marks a missing value.
    #[wasm_bindgen(js_name = addNumeric)]
    pub fn add_numeric(&mut self, values: Vec<f64>) -> Result<usize, JsError> {
        self.inner.add_dimension(Keys::Numeric(values)).map_err(js)
    }

    /// Adds a categorical dimension of codes in `0..cardinality`.
    #[wasm_bindgen(js_name = addCategorical)]
    pub fn add_categorical(&mut self, codes: Vec<u32>, cardinality: u32) -> Result<usize, JsError> {
        self.inner.add_dimension(Keys::Categorical { codes, cardinality }).map_err(js)
    }

    /// `[min, max]` of a numeric dimension's finite values, or `undefined` if it has none.
    pub fn extent(&self, dim: usize) -> Result<Option<Vec<f64>>, JsError> {
        Ok(self.inner.extent(dim).map_err(js)?.map(|(lo, hi)| vec![lo, hi]))
    }

    /// Returns how many rows crossed the filter boundary.
    #[wasm_bindgen(js_name = filterRange)]
    pub fn filter_range(&mut self, dim: usize, lo: f64, hi: f64) -> Result<usize, JsError> {
        self.inner.filter_range(dim, lo, hi).map_err(js)
    }

    #[wasm_bindgen(js_name = filterCodes)]
    pub fn filter_codes(&mut self, dim: usize, codes: Vec<u32>) -> Result<usize, JsError> {
        self.inner.filter_codes(dim, &codes).map_err(js)
    }

    #[wasm_bindgen(js_name = clearFilter)]
    pub fn clear_filter(&mut self, dim: usize) -> Result<usize, JsError> {
        self.inner.clear_filter(dim).map_err(js)
    }

    /// Equal-width histogram. Omit `min`/`max` to use the dimension's extent.
    #[wasm_bindgen(js_name = addHistogram)]
    pub fn add_histogram(&mut self, dim: usize, bins: u32, min: Option<f64>, max: Option<f64>) -> Result<usize, JsError> {
        let a = axis(dim, bins, min, max);
        self.inner.add_group(Binning::Histogram { dim, min: a.min, width: a.width, bins }).map_err(js)
    }

    #[wasm_bindgen(js_name = addCategoryCounts)]
    pub fn add_category_counts(&mut self, dim: usize) -> Result<usize, JsError> {
        self.inner.add_group(Binning::Category { dim, cardinality: 0 }).map_err(js)
    }

    /// `xBins * yBins` density grid over two numeric dimensions, row-major in y.
    #[wasm_bindgen(js_name = addGrid)]
    pub fn add_grid(&mut self, x: usize, y: usize, x_bins: u32, y_bins: u32) -> Result<usize, JsError> {
        self.inner
            .add_group(Binning::Grid { x: axis(x, x_bins, None, None), y: axis(y, y_bins, None, None) })
            .map_err(js)
    }

    #[wasm_bindgen(js_name = groupCounts)]
    pub fn group_counts(&self, group: usize) -> Result<Vec<u32>, JsError> {
        Ok(self.inner.group_counts(group).map_err(js)?.to_vec())
    }

    /// Bin edges as `[min, width, bins]` (histogram), `[cardinality]` (category), or
    /// `[xMin, xWidth, xBins, yMin, yWidth, yBins]` (grid).
    #[wasm_bindgen(js_name = groupBinning)]
    pub fn group_binning(&self, group: usize) -> Result<Vec<f64>, JsError> {
        Ok(match self.inner.group_binning(group).map_err(js)? {
            Binning::Histogram { min, width, bins, .. } => vec![min, width, bins as f64],
            Binning::Category { cardinality, .. } => vec![cardinality as f64],
            Binning::Grid { x, y } => vec![x.min, x.width, x.bins as f64, y.min, y.width, y.bins as f64],
        })
    }

    #[wasm_bindgen(js_name = selectedCount)]
    pub fn selected_count(&self) -> usize {
        self.inner.selected_count()
    }

    /// Pointer to `rows()` bytes: 1 where the row passes every filter.
    #[wasm_bindgen(js_name = selectedPtr)]
    pub fn selected_ptr(&self) -> *const u8 {
        self.inner.selected().as_ptr()
    }

    /// Pointer to `changedLen()` row ids whose selected state flipped in the last change.
    #[wasm_bindgen(js_name = changedPtr)]
    pub fn changed_ptr(&self) -> *const u32 {
        self.inner.changed().as_ptr()
    }

    #[wasm_bindgen(js_name = changedLen)]
    pub fn changed_len(&self) -> usize {
        self.inner.changed().len()
    }

    #[wasm_bindgen(js_name = setScanThreshold)]
    pub fn set_scan_threshold(&mut self, rows: usize) {
        self.inner.set_scan_threshold(rows);
    }
}

/// The module's linear memory, for building zero-copy views.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// NaN asks chigutiro for the dimension's extent.
fn axis(dim: usize, bins: u32, min: Option<f64>, max: Option<f64>) -> Axis {
    match (min, max) {
        (Some(lo), Some(hi)) if bins > 0 => Axis { dim, min: lo, width: (hi - lo) / bins as f64, bins },
        _ => Axis { dim, min: f64::NAN, width: f64::NAN, bins },
    }
}
