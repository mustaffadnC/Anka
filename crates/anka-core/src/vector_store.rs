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
    /// Backed by a snapshot mapping and read without copying (used from phase 3).
    Mapped { map: Mmap, offset: usize },
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

    /// Whether this store reads through a memory mapping rather than owning its buffer.
    pub fn is_mapped(&self) -> bool {
        matches!(self.storage, Storage::Mapped { .. })
    }

    /// Size of the vector data in bytes.
    ///
    /// For the mapped variant this is the size of the mapped region — address space, not
    /// necessarily resident memory. The distinction is exactly what the quantization memory
    /// table in `docs/RESULTS.md` turns on, so the two are never conflated.
    pub fn data_bytes(&self) -> usize {
        self.count * self.dim * size_of::<f32>()
    }

    /// The entire buffer as one flat slice.
    ///
    /// Hoist this out of hot loops instead of calling [`Self::get`] per candidate: for the
    /// mapped variant it repeats an alignment check the constructor already guaranteed.
    /// Whether removing that check with a documented `unsafe` is worth it is a question for
    /// phase 2, when there is a profile to answer it with.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        match &self.storage {
            Storage::Owned(data) => data,
            Storage::Mapped { map, offset } => {
                let end = offset + self.count * self.dim * size_of::<f32>();
                bytemuck::cast_slice(&map[*offset..end])
            }
        }
    }

    /// The vector at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= self.len()`. Use [`Self::try_get`] where the index comes from outside.
    #[inline]
    pub fn get(&self, index: usize) -> &[f32] {
        let start = index * self.dim;
        &self.as_slice()[start..start + self.dim]
    }

    #[inline]
    pub fn try_get(&self, index: usize) -> Option<&[f32]> {
        (index < self.count).then(|| self.get(index))
    }

    /// Iterates over the vectors in storage order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f32]> {
        self.as_slice().chunks_exact(self.dim)
    }

    /// Appends a vector. Owned storage only.
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

        match &mut self.storage {
            Storage::Owned(data) => {
                data.extend_from_slice(vector);
                self.count += 1;
                Ok(())
            }
            Storage::Mapped { .. } => Err(VectorError::ReadOnlyStorage),
        }
    }

    /// Scans for NaN and infinity. See [`Self::from_mmap`] for why this is not automatic
    /// there.
    pub fn validate_finite(&self) -> Result<(), VectorError> {
        validate_finite(self.as_slice(), self.dim, 0)
    }
}

impl std::fmt::Debug for VectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorStore")
            .field("dim", &self.dim)
            .field("count", &self.count)
            .field(
                "storage",
                &if self.is_mapped() { "mapped" } else { "owned" },
            )
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
        assert_eq!(store.as_slice(), &values);
    }

    #[test]
    fn mapped_storage_is_read_only() {
        let mut store = mapped_store(128, 2, &[1.0, 2.0]).expect("valid mapping");
        assert!(matches!(
            store.push(&[3.0, 4.0]),
            Err(VectorError::ReadOnlyStorage)
        ));
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
