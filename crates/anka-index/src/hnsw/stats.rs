//! Distance-computation counting.
//!
//! This is the metric that tells an algorithmic win apart from a micro-optimisation, and phase 1
//! measured why it matters: a brute-force scan over SIFT1M runs at the DDR5 ceiling, so there is
//! nothing left to gain in the kernel. What an index changes is how many distances get computed
//! at all, and that is what this counts.
//!
//! It also sits in the hottest loop in the project, so it is behind the `stats` feature and
//! compiles to a zero-sized struct with an empty method when that feature is off. A build with
//! `stats` on does not produce publishable QPS numbers, and `docs/RESULTS.md` keeps the two kinds
//! of run apart.

/// Counts distance computations, or nothing at all.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DistanceCounter {
    #[cfg(feature = "stats")]
    count: u64,
}

impl DistanceCounter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `n` distance computations.
    #[inline(always)]
    pub fn record(&mut self, n: u64) {
        #[cfg(feature = "stats")]
        {
            self.count += n;
        }
        #[cfg(not(feature = "stats"))]
        {
            let _ = n;
        }
    }

    /// The count, or `None` when this build is not instrumented.
    ///
    /// `Option` rather than a bare `0`, so an uninstrumented build cannot be mistaken for a
    /// search that computed no distances.
    pub fn count(&self) -> Option<u64> {
        #[cfg(feature = "stats")]
        {
            Some(self.count)
        }
        #[cfg(not(feature = "stats"))]
        {
            None
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_only_when_instrumented() {
        let mut counter = DistanceCounter::new();
        counter.record(3);
        counter.record(1);

        if cfg!(feature = "stats") {
            assert_eq!(counter.count(), Some(4));
        } else {
            assert_eq!(counter.count(), None);
        }
    }

    #[test]
    fn reset_clears() {
        let mut counter = DistanceCounter::new();
        counter.record(10);
        counter.reset();
        assert_eq!(
            counter.count(),
            if cfg!(feature = "stats") {
                Some(0)
            } else {
                None
            }
        );
    }

    /// Without the feature the counter must not cost a byte, or it would not be safe to leave in
    /// the hot loop.
    #[test]
    fn is_zero_sized_without_the_feature() {
        let expected = if cfg!(feature = "stats") { 8 } else { 0 };
        assert_eq!(size_of::<DistanceCounter>(), expected);
    }
}
