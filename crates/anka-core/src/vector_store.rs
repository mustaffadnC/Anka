//! Flat, row-major storage for fp32 vectors.
//!
//! Vectors are laid out as `[v0_0..v0_d, v1_0..v1_d, ...]` in a single allocation, never as
//! `Vec<Vec<f32>>`: the latter turns every distance computation into pointer chasing and
//! destroys cache locality, which is the one thing this code cannot afford to get wrong.

use memmap2::Mmap;

use crate::error::VectorError;

/// Upper bound on dimensionality.
///
/// Nothing in the design needs a limit this high — the widest dataset in the benchmark suite
/// is 768 — but having one turns a corrupt header into an error instead of a multi-gigabyte
/// allocation.
pub const MAX_DIM: usize = 4096;

/// Where a [`VectorStore`]'s bytes live.
pub enum Storage {
    /// Built in memory: the insert and index-construction path.
    Owned(Vec<f32>),
    /// Backed by a snapshot mapping and read without copying.
    Mapped { map: Mmap, offset: usize },
    /// A snapshot mapping with vectors appended after it.
    ///
    /// This is what recovery produces. A snapshot is loaded read-only, then the write-ahead log
    /// is replayed on top of it, and those replayed vectors have nowhere to go inside a read-only
    /// mapping. The alternative was to copy the mapping into an owned buffer on the first write —
    /// half a gigabyte on SIFT1M, paid by any collection that is ever written to, which would
    /// defeat the entire point of mapping it.
    Hybrid {
        map: Mmap,
        offset: usize,
        /// Vectors `0..mapped_count` come from the mapping; the rest come from `owned`.
        mapped_count: usize,
        owned: Vec<f32>,
    },
}

/// A borrowed view over a store's vectors, with the storage variant already resolved.
///
/// Hot loops take one of these once and then index it, rather than matching on [`Storage`] per
/// access. `Contiguous` covers both owned and fully-mapped stores, so the common path — an index
/// built in memory, which is every measurement in `docs/RESULTS.md` — costs a single
/// well-predicted branch per vector.
#[derive(Clone, Copy)]
pub enum Vectors<'a> {
    Contiguous {
        data: &'a [f32],
        dim: usize,
    },
    Split {
        mapped: &'a [f32],
        owned: &'a [f32],
        dim: usize,
        /// First index served by `owned`.
        split: usize,
    },
}

impl<'a> Vectors<'a> {
    /// The vector at `index`.
    ///
    /// # Panics
    ///
    /// If `index` is out of range. Callers in the search path have already bounded their indices
    /// by the graph, which only ever holds ids that exist.
    #[inline]
    pub fn get(&self, index: usize) -> &'a [f32] {
        match *self {
            Self::Contiguous { data, dim } => {
                let start = index * dim;
                &data[start..start + dim]
            }
            Self::Split {
                mapped,
                owned,
                dim,
                split,
            } => {
                if index < split {
                    let start = index * dim;
                    &mapped[start..start + dim]
                } else {
                    let start = (index - split) * dim;
                    &owned[start..start + dim]
                }
            }
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        match *self {
            Self::Contiguous { dim, .. } | Self::Split { dim, .. } => dim,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match *self {
            Self::Contiguous { data, dim } => data.len() / dim,
            Self::Split {
                mapped, owned, dim, ..
            } => (mapped.len() + owned.len()) / dim,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over the vectors in storage order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a [f32]> + use<'a> {
        let view = *self;
        (0..view.len()).map(move |index| view.get(index))
    }
}

/// A collection of equal-length fp32 vectors.
pub struct VectorStore {
    dim: usize,
    count: usize,
    storage: Storage,
}

impl VectorStore {
    /// An empty store that vectors can be pushed into.
    pub fn empty(dim: usize) -> Result<Self, VectorError> {
        validate_dim(dim)?;
        Ok(Self {
            dim,
            count: 0,
            storage: Storage::Owned(Vec::new()),
        })
    }

    /// Takes ownership of an already-flat buffer.
    ///
    /// The data is validated for finiteness here: it is already resident and warm in cache,
    /// so the scan is cheap, and rejecting at the boundary is what lets the search path skip
    /// defensive checks entirely.
    pub fn from_flat(dim: usize, data: Vec<f32>) -> Result<Self, VectorError> {
        validate_dim(dim)?;
        if !data.len().is_multiple_of(dim) {
            return Err(VectorError::RaggedBuffer {
                len: data.len(),
                dim,
            });
        }
        let count = data.len() / dim;
        check_count(count)?;
        validate_finite(&data, dim, 0)?;
        Ok(Self {
            dim,
            count,
            storage: Storage::Owned(data),
        })
    }

    /// Views `count` vectors of `dim` components starting `offset` bytes into a mapping.
    ///
    /// Shape and alignment are checked, but **finiteness is not** — scanning the whole region
    /// would page in every byte and defeat the point of mapping it lazily. Call
    /// [`Self::validate_finite`] explicitly when that cost is acceptable. This mirrors how the
    /// snapshot format treats its two checksums: the cheap one always, the expensive one on
    /// request.
    pub fn from_mmap(
        map: Mmap,
        offset: usize,
        dim: usize,
        count: usize,
    ) -> Result<Self, VectorError> {
        validate_dim(dim)?;
        check_count(count)?;

        if !offset.is_multiple_of(align_of::<f32>()) {
            return Err(VectorError::MisalignedOffset { offset });
        }

        let needed = count
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .ok_or(VectorError::TooManyVectors { count })?;
        let available = map.len().saturating_sub(offset);
        if available < needed {
            return Err(VectorError::MappingTooSmall {
                dim,
                count,
                offset,
                needed,
                available,
            });
        }

        Ok(Self {
            dim,
            count,
            storage: Storage::Mapped { map, offset },
        })
    }

    /// Number of vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether any of this store reads through a memory mapping.
    pub fn is_mapped(&self) -> bool {
        matches!(
            self.storage,
            Storage::Mapped { .. } | Storage::Hybrid { .. }
        )
    }

    /// How many vectors come from a mapping rather than from owned memory.
    pub fn mapped_count(&self) -> usize {
        match &self.storage {
            Storage::Owned(_) => 0,
            Storage::Mapped { .. } => self.count,
            Storage::Hybrid { mapped_count, .. } => *mapped_count,
        }
    }

    /// Bytes this store owns outright.
    ///
    /// Distinct from [`Self::data_bytes`], which counts the whole collection including whatever is
    /// only mapped. Phase 5's memory claim is about resident bytes specifically, so the two are
    /// separate methods rather than one number that has to be explained.
    pub fn resident_bytes(&self) -> usize {
        match &self.storage {
            Storage::Owned(data) => data.capacity() * size_of::<f32>(),
            Storage::Mapped { .. } => 0,
            Storage::Hybrid { owned, .. } => owned.capacity() * size_of::<f32>(),
        }
    }

    /// Size of the vector data in bytes.
    ///
    /// For the mapped variant this is the size of the mapped region — address space, not
    /// necessarily resident memory. The distinction is exactly what the quantization memory
    /// table in `docs/RESULTS.md` turns on, so the two are never conflated.
    pub fn data_bytes(&self) -> usize {
        self.count * self.dim * size_of::<f32>()
    }

    /// A view with the storage variant resolved. Hoist this out of hot loops.
    #[inline]
    pub fn view(&self) -> Vectors<'_> {
        match &self.storage {
            Storage::Owned(data) => Vectors::Contiguous {
                data,
                dim: self.dim,
            },
            Storage::Mapped { map, offset } => Vectors::Contiguous {
                data: mapped_slice(map, *offset, self.count, self.dim),
                dim: self.dim,
            },
            Storage::Hybrid {
                map,
                offset,
                mapped_count,
                owned,
            } => Vectors::Split {
                mapped: mapped_slice(map, *offset, *mapped_count, self.dim),
                owned,
                dim: self.dim,
                split: *mapped_count,
            },
        }
    }

    /// The whole collection as one flat slice, when it happens to be contiguous.
    ///
    /// `None` for a hybrid store, where the vectors live in two places by construction. Callers
    /// that genuinely need one buffer — writing a snapshot, slicing a prefix — handle the `None`;
    /// callers that only need to read vectors should use [`Self::view`] instead and never care.
    #[inline]
    pub fn as_contiguous(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::Owned(data) => Some(data),
            Storage::Mapped { map, offset } => {
                Some(mapped_slice(map, *offset, self.count, self.dim))
            }
            Storage::Hybrid { .. } => None,
        }
    }

    /// The entire buffer as one mutable flat slice.
    ///
    /// Owned storage only. A mapping is opened read-only, and a hybrid store is only half
    /// writable — normalising one in place would silently skip the mapped half, which is exactly
    /// the kind of half-applied transformation that produces a plausible, wrong recall figure.
    /// Used by [`crate::preprocess_all`] to normalise without copying half a gigabyte.
    pub fn as_mut_slice(&mut self) -> Result<&mut [f32], VectorError> {
        match &mut self.storage {
            Storage::Owned(data) => Ok(data),
            Storage::Mapped { .. } | Storage::Hybrid { .. } => Err(VectorError::ReadOnlyStorage),
        }
    }

    /// The vector at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`. Use [`Self::try_get`] where the index comes from outside, and
    /// [`Self::view`] in a loop — this resolves the storage variant on every call.
    #[inline]
    pub fn get(&self, index: usize) -> &[f32] {
        self.view().get(index)
    }

    #[inline]
    pub fn try_get(&self, index: usize) -> Option<&[f32]> {
        (index < self.count).then(|| self.get(index))
    }

    /// Iterates over the vectors in storage order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f32]> {
        self.view().iter()
    }

    /// Appends a vector.
    ///
    /// A fully mapped store becomes hybrid on the first push: the mapping stays where it is and
    /// the new vector goes into a fresh owned buffer beside it. That transition is what makes
    /// "load a snapshot, then replay the log on top of it" possible without copying the snapshot.
    pub fn push(&mut self, vector: &[f32]) -> Result<(), VectorError> {
        if vector.len() != self.dim {
            return Err(VectorError::DimMismatch {
                expected: self.dim,
                found: vector.len(),
            });
        }
        if let Some(component) = vector.iter().position(|v| !v.is_finite()) {
            return Err(VectorError::NonFinite {
                vector: self.count,
                component,
                value: vector[component],
            });
        }
        check_count(self.count + 1)?;

        // `take` so the mapping can be moved into the new variant; the store is left momentarily
        // holding an empty owned buffer, which nothing can observe from inside this method.
        let storage = std::mem::replace(&mut self.storage, Storage::Owned(Vec::new()));
        self.storage = match storage {
            Storage::Owned(mut data) => {
                data.extend_from_slice(vector);
                Storage::Owned(data)
            }
            Storage::Mapped { map, offset } => Storage::Hybrid {
                map,
                offset,
                mapped_count: self.count,
                owned: vector.to_vec(),
            },
            Storage::Hybrid {
                map,
                offset,
                mapped_count,
                mut owned,
            } => {
                owned.extend_from_slice(vector);
                Storage::Hybrid {
                    map,
                    offset,
                    mapped_count,
                    owned,
                }
            }
        };
        self.count += 1;
        Ok(())
    }

    /// Scans for NaN and infinity. See [`Self::from_mmap`] for why this is not automatic
    /// there.
    pub fn validate_finite(&self) -> Result<(), VectorError> {
        match self.as_contiguous() {
            Some(data) => validate_finite(data, self.dim, 0),
            None => {
                let view = self.view();
                for index in 0..self.count {
                    validate_finite(view.get(index), self.dim, index)?;
                }
                Ok(())
            }
        }
    }
}

/// Reinterprets `count` vectors starting `offset` bytes into a mapping.
///
/// The constructor has already checked alignment and size, so `cast_slice` cannot fail here.
#[inline]
fn mapped_slice(map: &Mmap, offset: usize, count: usize, dim: usize) -> &[f32] {
    let end = offset + count * dim * size_of::<f32>();
    bytemuck::cast_slice(&map[offset..end])
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("dim", &self.dim)
            .field("count", &self.count)
            .field(
                "storage",
                &match &self.storage {
                    Storage::Owned(_) => "owned",
                    Storage::Mapped { .. } => "mapped",
                    Storage::Hybrid { .. } => "hybrid",
                },
            )
            .field("mapped_count", &self.mapped_count())
            .finish()
    }
}

pub(crate) fn validate_dim(dim: usize) -> Result<(), VectorError> {
    if dim == 0 {
        return Err(VectorError::ZeroDim);
    }
    if dim > MAX_DIM {
        return Err(VectorError::DimTooLarge { dim, max: MAX_DIM });
    }
    Ok(())
}

/// `NodeId` is a `u32`, so the graph cannot address more vectors than that.
fn check_count(count: usize) -> Result<(), VectorError> {
    if count > u32::MAX as usize {
        return Err(VectorError::TooManyVectors { count });
    }
    Ok(())
}

/// One linear pass; roughly 130 M checks for SIFT1M, tens of milliseconds, done once.
/// `vector_base` lets callers report indices relative to a larger collection.
fn validate_finite(data: &[f32], dim: usize, vector_base: usize) -> Result<(), VectorError> {
    match data.iter().position(|v| !v.is_finite()) {
        Some(flat) => Err(VectorError::NonFinite {
            vector: vector_base + flat / dim,
            component: flat % dim,
            value: data[flat],
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn store_of(dim: usize, data: Vec<f32>) -> VectorStore {
        VectorStore::from_flat(dim, data).expect("valid store")
    }

    #[test]
    fn from_flat_exposes_rows() {
        let store = store_of(2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(store.dim(), 2);
        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());
        assert_eq!(store.get(0), &[1.0, 2.0]);
        assert_eq!(store.get(2), &[5.0, 6.0]);
        assert_eq!(store.data_bytes(), 6 * 4);
        assert!(!store.is_mapped());
    }

    #[test]
    fn iter_yields_every_row_in_order() {
        let store = store_of(2, vec![1.0, 2.0, 3.0, 4.0]);
        let rows: Vec<&[f32]> = store.iter().collect();
        assert_eq!(rows, vec![&[1.0, 2.0][..], &[3.0, 4.0][..]]);
    }

    #[test]
    fn try_get_is_bounded() {
        let store = store_of(2, vec![1.0, 2.0]);
        assert_eq!(store.try_get(0), Some(&[1.0, 2.0][..]));
        assert_eq!(store.try_get(1), None);
        assert_eq!(store.try_get(usize::MAX), None);
    }

    #[test]
    fn zero_dimension_is_rejected() {
        assert!(matches!(
            VectorStore::from_flat(0, vec![]),
            Err(VectorError::ZeroDim)
        ));
    }

    #[test]
    fn oversized_dimension_is_rejected() {
        let dim = MAX_DIM + 1;
        assert!(matches!(
            VectorStore::from_flat(dim, vec![0.0; dim]),
            Err(VectorError::DimTooLarge { .. })
        ));
    }

    #[test]
    fn ragged_buffer_is_rejected() {
        assert!(matches!(
            VectorStore::from_flat(3, vec![1.0, 2.0, 3.0, 4.0]),
            Err(VectorError::RaggedBuffer { len: 4, dim: 3 })
        ));
    }

    #[test]
    fn empty_store_is_valid() {
        let store = VectorStore::empty(4).expect("valid dim");
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.iter().count(), 0);
    }

    /// The reported vector and component indices are the whole point of this error: a bare
    /// "contains NaN" on a million vectors is not actionable.
    #[test]
    fn non_finite_values_are_rejected_with_a_location() {
        let err = VectorStore::from_flat(3, vec![1.0, 2.0, 3.0, 4.0, f32::NAN, 6.0])
            .expect_err("NaN must be rejected");
        match err {
            VectorError::NonFinite {
                vector,
                component,
                value,
            } => {
                assert_eq!((vector, component), (1, 1));
                assert!(value.is_nan());
            }
            other => panic!("expected NonFinite, got {other:?}"),
        }

        for bad in [f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                VectorStore::from_flat(1, vec![bad]),
                Err(VectorError::NonFinite { .. })
            ));
        }
    }

    #[test]
    fn push_appends_and_validates() {
        let mut store = VectorStore::empty(2).unwrap();
        store.push(&[1.0, 2.0]).expect("matching dim");
        store.push(&[3.0, 4.0]).expect("matching dim");
        assert_eq!(store.len(), 2);
        assert_eq!(store.get(1), &[3.0, 4.0]);

        assert!(matches!(
            store.push(&[1.0]),
            Err(VectorError::DimMismatch {
                expected: 2,
                found: 1
            })
        ));
        assert!(matches!(
            store.push(&[1.0, f32::NAN]),
            Err(VectorError::NonFinite { component: 1, .. })
        ));
        // A rejected push must not have moved the store forward.
        assert_eq!(store.len(), 2);
    }

    /// Writes a flat f32 buffer at `offset` bytes into a file and maps it, which is the shape
    /// the snapshot reader will use in phase 3.
    fn mapped_store(offset: usize, dim: usize, values: &[f32]) -> Result<VectorStore, VectorError> {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&vec![0u8; offset]).expect("padding");
        file.write_all(bytemuck::cast_slice(values)).expect("data");
        file.flush().expect("flush");

        // SAFETY: memory mapping is inherently unsafe because another process could truncate
        // the file underneath us. This is a temp file created and held by this test, so no
        // other writer exists. (Permitted use of `unsafe` per the project rules: mmap.)
        let map = unsafe { Mmap::map(file.as_file()) }.expect("map");
        VectorStore::from_mmap(map, offset, dim, values.len() / dim)
    }

    #[test]
    fn mapped_storage_reads_without_copying() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let store = mapped_store(128, 3, &values).expect("valid mapping");
        assert!(store.is_mapped());
        assert_eq!(store.len(), 2);
        assert_eq!(store.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(store.get(1), &[4.0, 5.0, 6.0]);
        assert_eq!(store.as_contiguous().unwrap(), &values);
    }

    /// The transition that makes recovery possible: a snapshot is mapped read-only, then the
    /// write-ahead log is replayed on top of it. Pushing does not copy the mapping — it opens an
    /// owned buffer beside it.
    #[test]
    fn pushing_to_a_mapped_store_makes_it_hybrid() {
        let mut store = mapped_store(128, 2, &[1.0, 2.0, 3.0, 4.0]).expect("valid mapping");
        assert_eq!(store.mapped_count(), 2);
        assert_eq!(store.resident_bytes(), 0);

        store.push(&[5.0, 6.0]).expect("push onto a mapping");

        assert_eq!(store.len(), 3);
        assert_eq!(store.mapped_count(), 2, "the mapping was not copied");
        assert!(store.resident_bytes() > 0, "the new vector is resident");
        assert!(store.is_mapped());
    }

    /// Reads have to cross the split without noticing it.
    #[test]
    fn a_hybrid_store_reads_across_the_split() {
        let mut store = mapped_store(128, 2, &[1.0, 2.0, 3.0, 4.0]).expect("valid mapping");
        store.push(&[5.0, 6.0]).unwrap();
        store.push(&[7.0, 8.0]).unwrap();

        // Last mapped, first owned, and everything either side.
        assert_eq!(store.get(0), &[1.0, 2.0]);
        assert_eq!(store.get(1), &[3.0, 4.0]);
        assert_eq!(store.get(2), &[5.0, 6.0]);
        assert_eq!(store.get(3), &[7.0, 8.0]);
        assert_eq!(store.try_get(4), None);

        let rows: Vec<&[f32]> = store.iter().collect();
        assert_eq!(
            rows,
            vec![
                &[1.0, 2.0][..],
                &[3.0, 4.0][..],
                &[5.0, 6.0][..],
                &[7.0, 8.0][..]
            ]
        );

        let view = store.view();
        assert_eq!(view.len(), 4);
        assert_eq!(view.dim(), 2);
        assert_eq!(view.get(2), &[5.0, 6.0]);
    }

    /// A hybrid store is two buffers by construction, so there is no single slice to hand out.
    #[test]
    fn a_hybrid_store_has_no_contiguous_slice() {
        let mut store = mapped_store(128, 2, &[1.0, 2.0]).expect("valid mapping");
        assert!(store.as_contiguous().is_some());
        store.push(&[3.0, 4.0]).unwrap();
        assert!(store.as_contiguous().is_none());
    }

    /// Normalising in place would silently skip the mapped half, which is the kind of
    /// half-applied transformation that yields a plausible, wrong recall figure.
    #[test]
    fn neither_mapped_nor_hybrid_storage_is_mutable() {
        let mut store = mapped_store(128, 2, &[1.0, 2.0]).expect("valid mapping");
        assert!(matches!(
            store.as_mut_slice(),
            Err(VectorError::ReadOnlyStorage)
        ));

        store.push(&[3.0, 4.0]).unwrap();
        assert!(matches!(
            store.as_mut_slice(),
            Err(VectorError::ReadOnlyStorage)
        ));
    }

    /// Two different paths through `validate_finite`, so both are exercised.
    ///
    /// A hybrid store has no contiguous slice, so the check walks it vector by vector — that path
    /// is covered here by a clean store. The reported-index behaviour is covered on a mapped store
    /// instead, because `push` refuses a non-finite vector and there is no way to get one into the
    /// owned half without bypassing it.
    #[test]
    fn finiteness_is_checked_on_both_storage_paths() {
        let mut hybrid = mapped_store(128, 2, &[1.0, 2.0, 3.0, 4.0]).expect("valid mapping");
        hybrid.push(&[5.0, 6.0]).unwrap();
        assert!(
            hybrid.as_contiguous().is_none(),
            "exercising the split path"
        );
        assert!(hybrid.validate_finite().is_ok());

        let bad = mapped_store(0, 2, &[1.0, 2.0, f32::NAN, 4.0]).expect("shape is valid");
        match bad.validate_finite() {
            Err(VectorError::NonFinite {
                vector, component, ..
            }) => assert_eq!((vector, component), (1, 0)),
            other => panic!("expected NonFinite at vector 1, got {other:?}"),
        }
    }

    #[test]
    fn push_rejects_bad_input_on_a_hybrid_store_too() {
        // One mapped vector, then one pushed: the store is hybrid and holds two.
        let mut store = mapped_store(128, 2, &[1.0, 2.0]).expect("valid mapping");
        store.push(&[3.0, 4.0]).unwrap();
        assert_eq!(store.len(), 2);

        assert!(matches!(
            store.push(&[1.0]),
            Err(VectorError::DimMismatch { .. })
        ));
        assert!(matches!(
            store.push(&[1.0, f32::NAN]),
            Err(VectorError::NonFinite { component: 1, .. })
        ));
        assert_eq!(store.len(), 2, "a rejected push must not advance the store");
    }

    #[test]
    fn misaligned_mapping_offset_is_rejected() {
        assert!(matches!(
            mapped_store(3, 2, &[1.0, 2.0]),
            Err(VectorError::MisalignedOffset { offset: 3 })
        ));
    }

    /// A snapshot whose header claims more vectors than the file holds must be an error, not
    /// an out-of-bounds read.
    #[test]
    fn mapping_smaller_than_the_declared_shape_is_rejected() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(bytemuck::cast_slice(&[1.0f32, 2.0]))
            .expect("data");
        file.flush().expect("flush");
        // SAFETY: as above — a temp file owned by this test, no concurrent writer.
        let map = unsafe { Mmap::map(file.as_file()) }.expect("map");

        assert!(matches!(
            VectorStore::from_mmap(map, 0, 2, 100),
            Err(VectorError::MappingTooSmall { .. })
        ));
    }

    #[test]
    fn mapped_finiteness_is_checked_on_request() {
        let ok = mapped_store(0, 2, &[1.0, 2.0]).expect("valid mapping");
        assert!(ok.validate_finite().is_ok());

        // from_mmap accepts this: validating would page in the whole region.
        let bad = mapped_store(0, 2, &[1.0, f32::NAN]).expect("shape is valid");
        assert!(matches!(
            bad.validate_finite(),
            Err(VectorError::NonFinite { .. })
        ));
    }
}
