use std::ops::Range;

/// A set of ordered types.
///
/// Implemented as an ordered list of ranges that do not touch or overlap.
#[derive(Clone, Debug, Default)]
pub struct RangeSet<T>(Vec<Range<T>>);

impl<T: Ord + Copy> RangeSet<T> {
    /// Returns a new empty `RangeSet`.
    #[inline]
    pub const fn new() -> Self {
        RangeSet(Vec::new())
    }

    /// Returns whether all elements of `range` are contained in the set.
    #[inline]
    pub fn contains(&self, elements: Range<T>) -> bool {
        self.0
            .iter()
            .any(|range| range.start <= elements.end && range.end >= elements.end)
    }

    /// Removes all ranges from the set.
    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Inserts the elements of `range` into the set.
    pub fn insert(&mut self, elements: Range<T>) {
        // Find the first range that is not less than `elements`, and the first range that is
        // greater.
        let index_start = self
            .0
            .iter()
            .position(|range| range.end >= elements.start)
            .unwrap_or(self.0.len());
        let index_end = self
            .0
            .iter()
            .position(|range| range.start > elements.end)
            .unwrap_or(self.0.len());

        if index_start == index_end {
            // `elements` fits in between index_start - 1 and index_start.
            self.0.insert(index_start, elements);
        } else {
            // `elements` touches the ranges between index_start and index_end.
            // Expand `elements` with the touched ranges, then store it in the first.
            self.0[index_start] = self.0[index_start..index_end]
                .iter()
                .fold(elements, |Range { start, end }, range| {
                    start.min(range.start)..end.max(range.end)
                });
            // Delete the remaining elements.
            self.0.drain(index_start + 1..index_end);
        }
    }
}
