//! Build parameters and layer assignment.

// rand 0.10 renamed things: `RngCore` became `Rng`, and the old `Rng` extension trait — the one
// carrying `random()` — became `RngExt`.
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::error::{IndexError, MAX_M};
use crate::hnsw::select::SelectionPolicy;

/// Upper bound on the level a node can be assigned.
///
/// A guard, not an expected value. With `M = 16` the multiplier is `1/ln(16) ≈ 0.361`, and a
/// level of 32 would need `u ≤ e^-88`, far below the smallest value a 53-bit uniform draw can
/// produce (~1.1e-16, which tops out around level 13). It exists so that a different `M` or a
/// future generator cannot produce a level that allocates thousands of empty layers.
pub const MAX_LEVEL: usize = 32;

/// How an HNSW graph is built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnswParams {
    /// Connections a new node makes per layer.
    m: usize,
    /// Cap on layer 0's degree. Conventionally `2M`: layer 0 carries every node and all the
    /// short-range structure, so it needs the extra room.
    max_degree0: usize,
    /// Candidate list size during construction. Larger means a better graph and a slower build.
    ef_construction: usize,
    /// Seed for layer assignment. Fixed by default — see `docs/DESIGN.md`, section 11.
    seed: u64,
    /// How neighbours are chosen. Configurable so the ablations the phase 2 DoD requires can be
    /// run by changing a flag rather than by editing the algorithm.
    selection: SelectionPolicy,
    /// `mL` in the paper: `1 / ln(M)`, which makes `P(level ≥ l) = M^-l`.
    level_multiplier: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        // The paper's defaults, and the ones the spec fixes for the published measurements.
        Self::new(16).expect("M = 16 is valid")
    }
}

impl HnswParams {
    /// Defaults derived from `m`: `max_degree0 = 2m`, `ef_construction = 200`, `seed = 0`.
    pub fn new(m: usize) -> Result<Self, IndexError> {
        Self {
            m,
            max_degree0: 2 * m,
            ef_construction: 200,
            seed: 0,
            selection: SelectionPolicy::default(),
            level_multiplier: 0.0,
        }
        .finish()
    }

    /// Overrides the neighbour-selection policy. Used for the heuristic and keep-pruned
    /// ablations; the default is what every published measurement uses.
    pub fn with_selection(mut self, selection: SelectionPolicy) -> Result<Self, IndexError> {
        self.selection = selection;
        self.finish()
    }

    pub fn with_max_degree0(mut self, max_degree0: usize) -> Result<Self, IndexError> {
        self.max_degree0 = max_degree0;
        self.finish()
    }

    pub fn with_ef_construction(mut self, ef_construction: usize) -> Result<Self, IndexError> {
        self.ef_construction = ef_construction;
        self.finish()
    }

    pub fn with_seed(mut self, seed: u64) -> Result<Self, IndexError> {
        self.seed = seed;
        self.finish()
    }

    fn finish(mut self) -> Result<Self, IndexError> {
        if self.m == 0 {
            return Err(IndexError::ZeroM);
        }
        if self.m > MAX_M {
            return Err(IndexError::MTooLarge { m: self.m });
        }
        if self.max_degree0 < self.m {
            return Err(IndexError::MaxDegreeTooSmall {
                m: self.m,
                max_degree0: self.max_degree0,
            });
        }
        if self.ef_construction == 0 {
            return Err(IndexError::ZeroEfConstruction);
        }
        // M = 1 gives ln(1) = 0 and an infinite multiplier. Every node would land on level 0,
        // which is a flat graph — degenerate, but a legitimate thing to measure, so it is
        // handled rather than rejected.
        self.level_multiplier = if self.m == 1 {
            0.0
        } else {
            1.0 / (self.m as f64).ln()
        };
        Ok(self)
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub fn max_degree0(&self) -> usize {
        self.max_degree0
    }

    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn level_multiplier(&self) -> f64 {
        self.level_multiplier
    }

    pub fn selection(&self) -> SelectionPolicy {
        self.selection
    }

    /// `Mmax(lc)` from the paper: layer 0 gets `max_degree0`, every layer above it gets `M`.
    ///
    /// Spelled out because the paper leaves it implicit and it is a natural thing to get wrong —
    /// using `M` everywhere starves layer 0, and using `max_degree0` everywhere doubles the
    /// graph for nothing.
    #[inline]
    pub fn max_degree(&self, layer: usize) -> usize {
        if layer == 0 { self.max_degree0 } else { self.m }
    }

    /// A generator for this configuration's layer assignment.
    pub fn level_generator(&self) -> LevelGenerator {
        LevelGenerator::new(self.seed, self.level_multiplier)
    }

    /// A generator fast-forwarded past `drawn` levels — the state a snapshot records.
    pub fn restored_level_generator(&self, drawn: u64) -> LevelGenerator {
        LevelGenerator::restore(self.seed, self.level_multiplier, drawn)
    }
}

/// Draws the level a new node is inserted at.
///
/// `level = floor(-ln(U) · mL)` with `U` uniform, giving `P(level ≥ l) = M^-l`: about one node in
/// `M` reaches layer 1, one in `M²` reaches layer 2. That geometric thinning is what makes the
/// upper layers a coarse index over the lower ones.
///
/// Seeded deliberately. `rand::rng()` appears nowhere in this project — an index whose shape
/// depends on unrecorded randomness cannot be measured twice, and H6 asks for exactly that.
/// Reproducibility across machines rests on the committed `Cargo.lock`, since `StdRng`'s
/// algorithm is only stable within a major version of `rand`.
pub struct LevelGenerator {
    rng: StdRng,
    multiplier: f64,
    drawn: u64,
}

impl LevelGenerator {
    pub fn new(seed: u64, multiplier: f64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            multiplier,
            drawn: 0,
        }
    }

    /// Re-creates the state a generator reaches after `drawn` levels, by drawing them again.
    ///
    /// `StdRng` has no portable way to export its state, and a snapshot that recorded an opaque
    /// blob would be tied to `rand`'s internals. Re-drawing costs one `f64` per vector — a few
    /// milliseconds at a million — and is exact by construction.
    pub fn restore(seed: u64, multiplier: f64, drawn: u64) -> Self {
        let mut generator = Self::new(seed, multiplier);
        for _ in 0..drawn {
            generator.next_level();
        }
        generator
    }

    /// How many levels this generator has produced.
    ///
    /// Not the same as the number of vectors in the index, which is why it has to be recorded
    /// separately: WAL replay inserts at the level written in the record and never draws one.
    /// An index restored from a snapshot plus replay therefore has more nodes than draws, and
    /// deriving the count from either would put the generator out of step.
    pub fn drawn(&self) -> u64 {
        self.drawn
    }

    /// The level for the next node.
    pub fn next_level(&mut self) -> usize {
        // Counted before the early return so both paths agree, which is what lets `restore`
        // reproduce the state by calling this same function.
        self.drawn += 1;
        if self.multiplier == 0.0 {
            return 0;
        }
        // random::<f64>() yields [0, 1); flipping it to (0, 1] keeps ln() away from -inf, which
        // would otherwise turn one unlucky draw into an infinite level.
        let uniform = 1.0 - self.rng.random::<f64>();
        let level = (-uniform.ln() * self.multiplier).floor();
        (level as usize).min(MAX_LEVEL)
    }
}

impl std::fmt::Debug for LevelGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LevelGenerator")
            .field("multiplier", &self.multiplier)
            .field("drawn", &self.drawn)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_follow_the_paper() {
        let params = HnswParams::default();
        assert_eq!(params.m(), 16);
        assert_eq!(params.max_degree0(), 32);
        assert_eq!(params.ef_construction(), 200);
        assert_eq!(params.seed(), 0);
        assert!((params.level_multiplier() - 1.0 / 16f64.ln()).abs() < 1e-12);
    }

    /// The definition the paper leaves implicit.
    #[test]
    fn layer_zero_gets_the_larger_degree_cap() {
        let params = HnswParams::default();
        assert_eq!(params.max_degree(0), 32);
        assert_eq!(params.max_degree(1), 16);
        assert_eq!(params.max_degree(9), 16);
    }

    #[test]
    fn builders_revalidate() {
        let params = HnswParams::new(8)
            .unwrap()
            .with_ef_construction(64)
            .unwrap()
            .with_seed(42)
            .unwrap();
        assert_eq!(
            (params.m(), params.ef_construction(), params.seed()),
            (8, 64, 42)
        );
    }

    #[test]
    fn invalid_parameters_are_rejected() {
        assert!(matches!(HnswParams::new(0), Err(IndexError::ZeroM)));
        assert!(matches!(
            HnswParams::new(MAX_M + 1),
            Err(IndexError::MTooLarge { .. })
        ));
        assert!(matches!(
            HnswParams::new(16).unwrap().with_max_degree0(8),
            Err(IndexError::MaxDegreeTooSmall {
                m: 16,
                max_degree0: 8
            })
        ));
        assert!(matches!(
            HnswParams::new(16).unwrap().with_ef_construction(0),
            Err(IndexError::ZeroEfConstruction)
        ));
    }

    /// M = 1 makes ln(M) zero and the multiplier infinite. Handled as a flat graph rather than
    /// left to produce NaN levels.
    #[test]
    fn m_of_one_gives_a_flat_graph() {
        let params = HnswParams::new(1).unwrap();
        assert_eq!(params.level_multiplier(), 0.0);
        let mut levels = params.level_generator();
        for _ in 0..1000 {
            assert_eq!(levels.next_level(), 0);
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_levels() {
        let params = HnswParams::default();
        let first: Vec<usize> = (0..500)
            .map(|_| params.level_generator().next_level())
            .collect();
        let second: Vec<usize> = (0..500)
            .map(|_| params.level_generator().next_level())
            .collect();
        assert_eq!(first, second);

        let mut a = params.level_generator();
        let mut b = params.level_generator();
        let sequence_a: Vec<usize> = (0..500).map(|_| a.next_level()).collect();
        let sequence_b: Vec<usize> = (0..500).map(|_| b.next_level()).collect();
        assert_eq!(sequence_a, sequence_b);
    }

    #[test]
    fn a_different_seed_gives_a_different_sequence() {
        let a = HnswParams::default();
        let b = HnswParams::default().with_seed(1).unwrap();
        let mut ga = a.level_generator();
        let mut gb = b.level_generator();
        let sa: Vec<usize> = (0..2000).map(|_| ga.next_level()).collect();
        let sb: Vec<usize> = (0..2000).map(|_| gb.next_level()).collect();
        assert_ne!(sa, sb);
    }

    /// The distribution is the whole point: `P(level ≥ l) = M^-l`. If this is wrong the upper
    /// layers are either too dense to be a coarse index or too sparse to be reachable, and
    /// recall suffers for a reason invisible in the search code.
    #[test]
    fn levels_thin_out_geometrically() {
        let m = 16usize;
        let params = HnswParams::new(m).unwrap();
        let mut generator = params.level_generator();

        let samples = 400_000;
        let mut at_least = [0usize; 4];
        for _ in 0..samples {
            let level = generator.next_level();
            for (l, count) in at_least.iter_mut().enumerate() {
                if level >= l {
                    *count += 1;
                }
            }
        }

        assert_eq!(at_least[0], samples, "every node is on layer 0");

        for (l, count) in at_least.iter().enumerate().skip(1) {
            let observed = *count as f64 / samples as f64;
            let expected = (m as f64).powi(-(l as i32));
            // Three standard deviations of a binomial proportion, with a floor so the l = 3
            // case (expected 1/4096) is not asserted into noise.
            let sigma = (expected * (1.0 - expected) / samples as f64).sqrt();
            let tolerance = (3.0 * sigma).max(expected * 0.25);
            assert!(
                (observed - expected).abs() <= tolerance,
                "P(level >= {l}): observed {observed:.6}, expected {expected:.6}, \
                 tolerance {tolerance:.6}"
            );
        }
    }

    /// A snapshot records the number of draws, not the generator's internals, so restoring has to
    /// put it back on exactly the same sequence.
    #[test]
    fn a_restored_generator_continues_the_same_sequence() {
        let params = HnswParams::default();

        let mut original = params.level_generator();
        let consumed: Vec<usize> = (0..1_000).map(|_| original.next_level()).collect();
        let rest: Vec<usize> = (0..500).map(|_| original.next_level()).collect();
        assert_eq!(original.drawn(), 1_500);

        let mut restored = params.restored_level_generator(consumed.len() as u64);
        assert_eq!(restored.drawn(), 1_000);
        let continued: Vec<usize> = (0..500).map(|_| restored.next_level()).collect();

        assert_eq!(continued, rest);
        assert_eq!(restored.drawn(), original.drawn());
    }

    /// The flat-graph shortcut returns before touching the RNG. The counter still has to advance,
    /// or restoring an `M = 1` index would disagree with itself.
    #[test]
    fn draws_are_counted_even_on_the_flat_graph_path() {
        let params = HnswParams::new(1).unwrap();
        let mut generator = params.level_generator();
        for _ in 0..100 {
            generator.next_level();
        }
        assert_eq!(generator.drawn(), 100);
        assert_eq!(params.restored_level_generator(100).drawn(), 100);
    }

    #[test]
    fn levels_never_exceed_the_guard() {
        // A tiny multiplier is not reachable through HnswParams; constructed directly to prove
        // the clamp holds whatever the generator is handed.
        let mut generator = LevelGenerator::new(7, 1e9);
        for _ in 0..1000 {
            assert!(generator.next_level() <= MAX_LEVEL);
        }
    }
}
