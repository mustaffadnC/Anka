//! Readers and writers for the `.fvecs` / `.ivecs` formats.
//!
//! Layout, per record: a little-endian `u32` dimension followed by that many little-endian
//! `f32` (`.fvecs`) or `i32` (`.ivecs`) values. Repeating the dimension on every record is
//! redundant, but it is a cheap and early corruption signal, so it is checked rather than
//! skipped.
//!
//! Files are read record by record through a `BufReader` straight into the destination
//! buffer. Slurping the file first would double peak memory on SIFT1M — 516 MB of bytes plus
//! 512 MB of floats — and phase 0 reports peak RSS as a deliverable.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use crate::error::{DatasetError, VectorError};
use crate::vector_store::{VectorStore, validate_dim};

const BUFFER_BYTES: usize = 1 << 20;

/// A `.ivecs` file: `count` rows of `dim` 32-bit integers.
///
/// Ground truth has this shape — for SIFT1M, 10 000 rows holding the ids of the 100 nearest
/// neighbours of each query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntMatrix {
    dim: usize,
    count: usize,
    data: Vec<i32>,
}

impl IntMatrix {
    pub fn new(dim: usize, data: Vec<i32>) -> Result<Self, VectorError> {
        validate_dim(dim)?;
        if !data.len().is_multiple_of(dim) {
            return Err(VectorError::RaggedBuffer {
                len: data.len(),
                dim,
            });
        }
        let count = data.len() / dim;
        Ok(Self { dim, count, data })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// # Panics
    ///
    /// If `index >= self.len()`.
    #[inline]
    pub fn row(&self, index: usize) -> &[i32] {
        let start = index * self.dim;
        &self.data[start..start + self.dim]
    }

    #[inline]
    pub fn try_row(&self, index: usize) -> Option<&[i32]> {
        (index < self.count).then(|| self.row(index))
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[i32]> {
        self.data.chunks_exact(self.dim)
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.data
    }

    /// Size of the row data in bytes.
    pub fn data_bytes(&self) -> usize {
        self.count * self.dim * size_of::<i32>()
    }
}

/// Reads a `.fvecs` file into a [`VectorStore`].
pub fn read_fvecs(path: impl AsRef<Path>) -> Result<VectorStore, DatasetError> {
    let path = path.as_ref();
    let (dim, data) = read_records::<f32>(path, "fvecs")?;
    VectorStore::from_flat(dim, data).map_err(|source| DatasetError::Vector {
        path: path.to_owned(),
        source,
    })
}

/// Reads an `.ivecs` file into an [`IntMatrix`].
pub fn read_ivecs(path: impl AsRef<Path>) -> Result<IntMatrix, DatasetError> {
    let path = path.as_ref();
    let (dim, data) = read_records::<i32>(path, "ivecs")?;
    IntMatrix::new(dim, data).map_err(|source| DatasetError::Vector {
        path: path.to_owned(),
        source,
    })
}

/// Writes a flat buffer of `dim`-component vectors as `.fvecs`.
pub fn write_fvecs(path: impl AsRef<Path>, dim: usize, data: &[f32]) -> Result<(), DatasetError> {
    write_records(path.as_ref(), dim, data)
}

/// Writes a flat buffer of `dim`-component rows as `.ivecs`.
pub fn write_ivecs(path: impl AsRef<Path>, dim: usize, data: &[i32]) -> Result<(), DatasetError> {
    write_records(path.as_ref(), dim, data)
}

fn read_records<T: bytemuck::Pod>(
    path: &Path,
    format: &'static str,
) -> Result<(usize, Vec<T>), DatasetError> {
    let file = File::open(path).map_err(|e| DatasetError::io(path, e))?;
    let len = file
        .metadata()
        .map_err(|e| DatasetError::io(path, e))?
        .len();
    if len == 0 {
        return Err(DatasetError::Empty {
            path: path.to_owned(),
        });
    }

    let mut reader = BufReader::with_capacity(BUFFER_BYTES, file);

    // The first record's dimension sets the stride for the entire file.
    let dim = read_u32(&mut reader, path, 0)? as usize;
    validate_dim(dim).map_err(|source| DatasetError::Vector {
        path: path.to_owned(),
        source,
    })?;

    let record_bytes = size_of::<u32>() + dim * size_of::<T>();
    if !len.is_multiple_of(record_bytes as u64) {
        return Err(DatasetError::Ragged {
            path: path.to_owned(),
            len,
            record_bytes,
            dim,
            format,
        });
    }
    let count = (len / record_bytes as u64) as usize;

    reader.rewind().map_err(|e| DatasetError::io(path, e))?;

    let mut data = vec![T::zeroed(); count * dim];
    for (record, row) in data.chunks_exact_mut(dim).enumerate() {
        let declared = read_u32(&mut reader, path, record)? as usize;
        if declared != dim {
            return Err(DatasetError::InconsistentDim {
                path: path.to_owned(),
                record,
                expected: dim,
                found: declared,
            });
        }
        // Little-endian on disk, little-endian in memory (asserted at the crate root), so the
        // destination can be filled as raw bytes without an intermediate buffer.
        reader
            .read_exact(bytemuck::cast_slice_mut(row))
            .map_err(|e| eof_as_truncated(e, path, record))?;
    }

    Ok((dim, data))
}

fn write_records<T: bytemuck::Pod>(
    path: &Path,
    dim: usize,
    data: &[T],
) -> Result<(), DatasetError> {
    validate_dim(dim).map_err(|source| DatasetError::Vector {
        path: path.to_owned(),
        source,
    })?;
    if !data.len().is_multiple_of(dim) {
        return Err(DatasetError::Vector {
            path: path.to_owned(),
            source: VectorError::RaggedBuffer {
                len: data.len(),
                dim,
            },
        });
    }

    let file = File::create(path).map_err(|e| DatasetError::io(path, e))?;
    let mut writer = BufWriter::with_capacity(BUFFER_BYTES, file);
    let header = (dim as u32).to_le_bytes();

    for row in data.chunks_exact(dim) {
        writer
            .write_all(&header)
            .map_err(|e| DatasetError::io(path, e))?;
        writer
            .write_all(bytemuck::cast_slice(row))
            .map_err(|e| DatasetError::io(path, e))?;
    }
    writer.flush().map_err(|e| DatasetError::io(path, e))?;
    Ok(())
}

fn read_u32(reader: &mut impl Read, path: &Path, record: usize) -> Result<u32, DatasetError> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| eof_as_truncated(e, path, record))?;
    Ok(u32::from_le_bytes(buf))
}

fn eof_as_truncated(error: std::io::Error, path: &Path, record: usize) -> DatasetError {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        DatasetError::Truncated {
            path: path.to_owned(),
            record,
        }
    } else {
        DatasetError::io(path, error)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    fn temp_path(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join(name);
        (dir, path)
    }

    fn write_raw(path: &Path, bytes: &[u8]) {
        let mut f = File::create(path).expect("create");
        f.write_all(bytes).expect("write");
        f.flush().expect("flush");
    }

    #[test]
    fn fvecs_round_trip() {
        let (_dir, path) = temp_path("round.fvecs");
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();

        write_fvecs(&path, 4, &data).expect("write");
        let store = read_fvecs(&path).expect("read");

        assert_eq!(store.dim(), 4);
        assert_eq!(store.len(), 3);
        assert_eq!(store.as_slice(), data.as_slice());
        assert_eq!(store.get(2), &[4.0, 4.5, 5.0, 5.5]);
    }

    #[test]
    fn ivecs_round_trip() {
        let (_dir, path) = temp_path("round.ivecs");
        let data: Vec<i32> = (0..10).collect();

        write_ivecs(&path, 5, &data).expect("write");
        let gt = read_ivecs(&path).expect("read");

        assert_eq!(gt.dim(), 5);
        assert_eq!(gt.len(), 2);
        assert_eq!(gt.row(0), &[0, 1, 2, 3, 4]);
        assert_eq!(gt.row(1), &[5, 6, 7, 8, 9]);
        assert_eq!(gt.try_row(2), None);
    }

    /// Ground truth files carry negative values nowhere, but the format is signed and a
    /// reader that quietly mangles them would be a bug waiting for a different dataset.
    #[test]
    fn ivecs_preserves_negative_values() {
        let (_dir, path) = temp_path("neg.ivecs");
        let data = vec![-1, i32::MIN, 0, i32::MAX];
        write_ivecs(&path, 2, &data).expect("write");
        assert_eq!(read_ivecs(&path).expect("read").as_slice(), data.as_slice());
    }

    #[test]
    fn single_record_file_reads() {
        let (_dir, path) = temp_path("one.fvecs");
        write_fvecs(&path, 3, &[1.0, 2.0, 3.0]).expect("write");
        let store = read_fvecs(&path).expect("read");
        assert_eq!(store.len(), 1);
        assert_eq!(store.get(0), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn empty_file_is_an_error() {
        let (_dir, path) = temp_path("empty.fvecs");
        write_raw(&path, &[]);
        assert!(matches!(read_fvecs(&path), Err(DatasetError::Empty { .. })));
    }

    #[test]
    fn missing_file_is_an_error() {
        let (_dir, path) = temp_path("absent.fvecs");
        assert!(matches!(read_fvecs(&path), Err(DatasetError::Io { .. })));
    }

    /// Fewer than four bytes: not even the dimension can be read.
    #[test]
    fn file_shorter_than_a_header_is_truncated() {
        let (_dir, path) = temp_path("stub.fvecs");
        write_raw(&path, &[1, 0]);
        assert!(matches!(
            read_fvecs(&path),
            Err(DatasetError::Truncated { record: 0, .. })
        ));
    }

    /// The common real-world corruption: an interrupted download or a partial copy.
    #[test]
    fn truncated_file_is_reported_as_ragged() {
        let (_dir, path) = temp_path("cut.fvecs");
        write_fvecs(&path, 4, &[0.0; 12]).expect("write");

        let mut bytes = std::fs::read(&path).expect("read back");
        bytes.truncate(bytes.len() - 7);
        write_raw(&path, &bytes);

        match read_fvecs(&path) {
            Err(DatasetError::Ragged {
                record_bytes, dim, ..
            }) => {
                assert_eq!((record_bytes, dim), (4 + 4 * 4, 4));
            }
            other => panic!("expected Ragged, got {other:?}"),
        }
    }

    /// A per-record dimension that disagrees with the first one means the stride is wrong;
    /// continuing would produce a plausible-looking but entirely wrong dataset.
    #[test]
    fn inconsistent_record_dimension_is_rejected() {
        let (_dir, path) = temp_path("mixed.fvecs");
        let mut bytes = Vec::new();
        // record 0: dim 2
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(bytemuck::cast_slice(&[1.0f32, 2.0]));
        // record 1: claims dim 7, same byte length so the file size check still passes
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(bytemuck::cast_slice(&[3.0f32, 4.0]));
        write_raw(&path, &bytes);

        assert!(matches!(
            read_fvecs(&path),
            Err(DatasetError::InconsistentDim {
                record: 1,
                expected: 2,
                found: 7,
                ..
            })
        ));
    }

    #[test]
    fn zero_dimension_header_is_rejected() {
        let (_dir, path) = temp_path("zero.fvecs");
        write_raw(&path, &0u32.to_le_bytes());
        assert!(matches!(
            read_fvecs(&path),
            Err(DatasetError::Vector {
                source: VectorError::ZeroDim,
                ..
            })
        ));
    }

    /// A corrupt header must not turn into a multi-gigabyte allocation.
    #[test]
    fn absurd_dimension_header_is_rejected() {
        let (_dir, path) = temp_path("huge.fvecs");
        write_raw(&path, &u32::MAX.to_le_bytes());
        assert!(matches!(
            read_fvecs(&path),
            Err(DatasetError::Vector {
                source: VectorError::DimTooLarge { .. },
                ..
            })
        ));
    }

    #[test]
    fn non_finite_values_in_a_file_are_rejected() {
        let (_dir, path) = temp_path("nan.fvecs");
        write_fvecs(&path, 2, &[1.0, 2.0, f32::NAN, 4.0]).expect("write");
        assert!(matches!(
            read_fvecs(&path),
            Err(DatasetError::Vector {
                source: VectorError::NonFinite { vector: 1, .. },
                ..
            })
        ));
    }

    #[test]
    fn writing_a_ragged_buffer_is_rejected() {
        let (_dir, path) = temp_path("ragged.fvecs");
        assert!(matches!(
            write_fvecs(&path, 3, &[1.0, 2.0]),
            Err(DatasetError::Vector {
                source: VectorError::RaggedBuffer { len: 2, dim: 3 },
                ..
            })
        ));
    }

    /// Larger than the 1 MiB read buffer, so the buffered path is exercised across refills
    /// rather than only within a single fill.
    #[test]
    fn file_larger_than_the_read_buffer_round_trips() {
        let (_dir, path) = temp_path("big.fvecs");
        let dim = 128;
        let count = 4096; // 4096 * (4 + 512) bytes ~= 2 MiB
        let data: Vec<f32> = (0..dim * count).map(|i| (i % 251) as f32).collect();

        write_fvecs(&path, dim, &data).expect("write");
        let store = read_fvecs(&path).expect("read");

        assert_eq!(store.len(), count);
        assert_eq!(store.dim(), dim);
        assert_eq!(store.get(count - 1), &data[(count - 1) * dim..]);
    }
}
