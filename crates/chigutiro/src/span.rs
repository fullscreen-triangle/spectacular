//! Sets of positions in a dimension's sorted order, as sorted disjoint half-open spans.

/// Positions `[start, end)` in a dimension's sorted order.
pub type Span = (u32, u32);

/// Writes `a \ b` into `out`. Both inputs must be sorted, disjoint and free of empty spans.
pub(crate) fn difference(a: &[Span], b: &[Span], out: &mut Vec<Span>) {
    out.clear();
    let mut j = 0;
    for &(start, end) in a {
        let mut cur = start;
        while j < b.len() && b[j].1 <= cur {
            j += 1;
        }
        let mut k = j;
        while cur < end {
            match b.get(k) {
                Some(&(bs, be)) if bs < end => {
                    if bs > cur {
                        out.push((cur, bs));
                    }
                    cur = cur.max(be);
                    k += 1;
                }
                _ => {
                    out.push((cur, end));
                    break;
                }
            }
        }
    }
}

/// Appends `span` to a sorted list, merging it with the last span when they touch.
pub(crate) fn push_merged(spans: &mut Vec<Span>, span: Span) {
    if span.0 >= span.1 {
        return;
    }
    match spans.last_mut() {
        Some(last) if last.1 == span.0 => last.1 = span.1,
        _ => spans.push(span),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(spans: &[Span]) -> Vec<u32> {
        spans.iter().flat_map(|&(s, e)| s..e).collect()
    }

    #[test]
    fn difference_matches_set_semantics() {
        let cases: &[(&[Span], &[Span])] = &[
            (&[(0, 10)], &[(3, 5)]),
            (&[(0, 10)], &[(0, 10)]),
            (&[(0, 10)], &[]),
            (&[], &[(0, 4)]),
            (&[(2, 4), (6, 9), (12, 20)], &[(0, 3), (8, 13), (19, 25)]),
            (&[(5, 6)], &[(0, 5), (6, 7)]),
        ];
        let mut out = Vec::new();
        for (a, b) in cases {
            difference(a, b, &mut out);
            let bs = expand(b);
            let want: Vec<u32> = expand(a).into_iter().filter(|p| !bs.contains(p)).collect();
            assert_eq!(expand(&out), want, "a={a:?} b={b:?}");
            assert!(out.iter().all(|&(s, e)| s < e));
        }
    }
}
