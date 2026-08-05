//! Writing an index to disk and mapping it back.
//!
//! The body follows the header's six sections, each padded to [`SECTION_ALIGN`]:
//!
//! ```text
//! Vectors      count × dim × f32
//! NodeLevels   count × u8            highest layer each node reaches
//! Layers       layer 0: adjacency only — its slot index *is* the node id
//!              layer n: node_count u32, then node_count × NodeId, then adjacency
//! Tombstones   empty until phase 4
//! IdMap        empty until phase 4
//! Metadata     empty until phase 4
//! ```
//!
//! Adjacency goes to disk exactly as it sits in memory. That is the whole reason the graph holds
//! slot offsets rather than pointers: offsets survive a restart, addresses do not.
//!
//! **What is written and what is derived.** The `NodeId → slot` table is *not* stored — it is the
//! inverse of the slot list, and two copies of one fact can disagree. Neither are the section
//! sizes: every one is computable from the header, so the reader checks the sizes it finds against
//! the sizes it expects, and a disagreement is a corrupt file rather than something to accommodate.
//!
//! **Ordering.** The write is `tmp → fsync(file) → rename → fsync(directory)`. The last step is
//! the one that gets left out: `rename` is atomic, but the *directory entry* it creates is not
//! durable until the directory itself is synced, so a crash can leave the old file in place with
//! the new one fully written and invisible.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anka_core::{NodeId, VectorStore};
use anka_index::{HnswIndex, HnswParams, IndexParts, Layer, SelectionPolicy};
use memmap2::Mmap;

use crate::error::SnapshotError;
use crate::header::{
    HEADER_BYTES, HeaderFlags, SECTION_ALIGN, SECTION_COUNT, Section, SnapshotHeader, crc32,
};

/// How much of a snapshot to check on load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Verify {
    /// Header checksum only: 124 bytes, and it catches a truncated or foreign file immediately.
    #[default]
    Header,
    /// Also checksum the body. Scans the whole file — 700 MB on SIFT1M — which defeats the lazy
    /// mapping, so it is opt-in and used by `--verify`, the crash tests and CI.
    Body,
}

/// Writes `index` to `path`, atomically.
///
/// `wal_seq` is the last write-ahead-log record this snapshot already contains; recovery replays
/// everything after it. Phase 3c has no log yet, so callers pass 0.
///
/// On success `path` either holds the complete new snapshot or, if the machine died partway, the
/// previous one untouched. There is no state in between: the new bytes are built under a temporary
/// name and only become `path` once they are durable.
pub fn write(index: &HnswIndex, path: &Path, wal_seq: u64) -> Result<(), SnapshotError> {
    let layout = Layout::of(index);
    let tmp = temp_path(path);

    // A fresh file every time. Truncating an existing one would leave the previous snapshot's
    // tail behind if this write is shorter, and that tail would checksum as corruption later.
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)
        .map_err(|e| SnapshotError::io(&tmp, e))?;

    // The header carries the body's checksum, so it cannot be written until the body has been
    // hashed. Its space is reserved here and filled in below — and reserved *outside* the
    // checksumming writer, so that writer hashes exactly what it is handed and nothing has to
    // reason about where the header ended.
    file.write_all(&[0u8; HEADER_BYTES])
        .map_err(|e| SnapshotError::io(&tmp, e))?;

    let body_crc32 = {
        let mut out = CrcWriter::new(BufWriter::new(&mut file));
        write_body(&mut out, index, &layout).map_err(|e| SnapshotError::io(&tmp, e))?;
        out.finish().map_err(|e| SnapshotError::io(&tmp, e))?
    };

    let header = header_for(index, &layout, wal_seq, body_crc32);
    file.seek(SeekFrom::Start(0))
        .map_err(|e| SnapshotError::io(&tmp, e))?;
    file.write_all(&header.encode())
        .map_err(|e| SnapshotError::io(&tmp, e))?;

    // The order below is the point of this function.
    file.sync_all().map_err(|e| SnapshotError::io(&tmp, e))?;
    drop(file);
    std::fs::rename(&tmp, path).map_err(|e| SnapshotError::io(path, e))?;
    sync_parent_directory(path)?;
    Ok(())
}

/// Maps a snapshot and rebuilds the index from it.
///
/// The vectors are *not* copied: they stay in the mapping and are paged in on demand. Everything
/// else — levels, adjacency — is copied, because the in-memory graph owns its arrays and mutating
/// a read-only mapping is not a thing.
///
/// This returns almost immediately regardless of file size; the cost of touching the vectors is
/// deferred to the queries that touch them. [`read`] is the other end of that trade, and
/// `anka snapshot` measures both.
pub fn load(path: &Path, verify: Verify) -> Result<HnswIndex, SnapshotError> {
    let file = File::open(path).map_err(|e| SnapshotError::io(path, e))?;
    // Safety: the mapping is read-only and lives as long as the store built from it. A concurrent
    // writer could still change the bytes underneath us, which is why the snapshot is written
    // under a temporary name and renamed into place rather than modified where it lies.
    let map = unsafe { Mmap::map(&file) }.map_err(|e| SnapshotError::io(path, e))?;

    let parsed = parse(&map, verify)?;
    let vectors = VectorStore::from_mmap(
        map,
        HEADER_BYTES + parsed.vectors_start,
        parsed.dim,
        parsed.count,
    )?;
    parsed.into_index(vectors)
}

/// Reads and validates a snapshot's header without touching the rest of the file.
///
/// Recovery needs one field from it — `wal_seq`, which says where the log picks up — and reading
/// 128 bytes to get it beats mapping 620 MB.
pub fn header(path: &Path) -> Result<SnapshotHeader, SnapshotError> {
    use std::io::Read;

    let mut file = File::open(path).map_err(|e| SnapshotError::io(path, e))?;
    let mut bytes = [0u8; HEADER_BYTES];
    file.read_exact(&mut bytes).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            // The length is what matters here, not the read error's wording.
            let got = std::fs::metadata(path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            SnapshotError::TooShort {
                needed: HEADER_BYTES,
                got,
            }
        } else {
            SnapshotError::io(path, e)
        }
    })?;
    SnapshotHeader::decode(&bytes)
}

/// The last write-ahead-log record this snapshot already contains.
pub fn wal_seq(path: &Path) -> Result<u64, SnapshotError> {
    Ok(header(path)?.wal_seq)
}

/// Reads a snapshot into owned memory and rebuilds the index from it.
///
/// The alternative to [`load`], and the reason it exists: reading pays for every byte up front,
/// where mapping pays per page on first touch. Which is faster depends entirely on how much of the
/// collection a workload actually visits — a graph search visits very little of it, which is the
/// whole argument for mapping. Having both makes that an experiment rather than an assertion.
pub fn read(path: &Path, verify: Verify) -> Result<HnswIndex, SnapshotError> {
    let bytes = std::fs::read(path).map_err(|e| SnapshotError::io(path, e))?;

    let parsed = parse(&bytes, verify)?;
    let start = HEADER_BYTES + parsed.vectors_start;
    let floats: &[f32] = bytemuck::try_cast_slice(
        &bytes[start..start + parsed.count * parsed.dim * size_of::<f32>()],
    )
    .map_err(|reason| SnapshotError::SectionNotCastable {
        section: Section::Vectors.name(),
        element: "f32",
        reason,
    })?;
    let vectors = VectorStore::from_flat(parsed.dim, floats.to_vec())?;
    parsed.into_index(vectors)
}

/// Everything decoded from a snapshot except the vectors, which the two loaders obtain differently.
struct Parsed {
    header: SnapshotHeader,
    params: HnswParams,
    node_levels: Vec<u8>,
    layers: Vec<Layer>,
    /// Body-relative offset of the vectors section.
    vectors_start: usize,
    count: usize,
    dim: usize,
}

impl Parsed {
    fn into_index(self, vectors: VectorStore) -> Result<HnswIndex, SnapshotError> {
        Ok(HnswIndex::from_parts(IndexParts {
            vectors,
            params: self.params,
            metric: self.header.metric,
            layers: self.layers,
            node_levels: self.node_levels,
            entry_point: self.header.entry_point,
            max_layer: self.header.max_layer as usize,
            levels_drawn: self.header.levels_drawn,
        })?)
    }
}

fn parse(bytes: &[u8], verify: Verify) -> Result<Parsed, SnapshotError> {
    let header = SnapshotHeader::decode(bytes)?;
    let body_end = HEADER_BYTES as u64 + header.body_len;
    if (bytes.len() as u64) < body_end {
        return Err(SnapshotError::BodyTruncated {
            file_len: bytes.len() as u64,
            body_len: header.body_len,
            header_len: HEADER_BYTES,
        });
    }
    let body = &bytes[HEADER_BYTES..body_end as usize];

    if verify == Verify::Body {
        let computed = crc32(body);
        if computed != header.body_crc32 {
            return Err(SnapshotError::BodyChecksumMismatch {
                stored: header.body_crc32,
                computed,
            });
        }
    }

    let count = header.count as usize;
    let dim = header.dim as usize;
    let params = params_from(&header)?;

    let node_levels = section(body, &header, Section::NodeLevels, count)?.to_vec();
    let layers = read_layers(body, &header, &params, count)?;
    let vectors_start = section_start(&header, Section::Vectors, count * dim * size_of::<f32>())?;

    Ok(Parsed {
        header,
        params,
        node_levels,
        layers,
        vectors_start,
        count,
        dim,
    })
}

/// Where each section starts, and how long the body is.
///
/// Computed up front rather than recorded while writing, so the header can be filled in with a
/// single seek back to offset 0 instead of being buffered alongside 700 MB of body.
struct Layout {
    offsets: [u64; SECTION_COUNT],
    vectors: usize,
    node_levels: usize,
    layers: usize,
}

impl Layout {
    fn of(index: &HnswIndex) -> Self {
        let vectors = index.len() * index.dim() * size_of::<f32>();
        let node_levels = index.len();
        let layers: usize = index
            .layers()
            .iter()
            .enumerate()
            .map(|(lc, layer)| layer_bytes(lc, layer))
            .sum();

        let mut offsets = [0u64; SECTION_COUNT];
        offsets[Section::NodeLevels as usize] = align(vectors as u64);
        offsets[Section::Layers as usize] =
            align(offsets[Section::NodeLevels as usize] + node_levels as u64);
        // Phase 4 fills these; until then they are three empty ranges at the end of the body.
        let end = align(offsets[Section::Layers as usize] + layers as u64);
        for section in [Section::Tombstones, Section::IdMap, Section::Metadata] {
            offsets[section as usize] = end;
        }

        Self {
            offsets,
            vectors,
            node_levels,
            layers,
        }
    }

    fn body_len(&self) -> u64 {
        self.offsets[Section::Metadata as usize]
    }
}

fn layer_bytes(index: usize, layer: &Layer) -> usize {
    let adjacency = size_of_val(layer.raw_neighbors());
    if index == 0 {
        // Dense: the slot index is the node id, so there is nothing else to store.
        adjacency
    } else {
        size_of::<u32>() + size_of_val(layer.slot_nodes()) + adjacency
    }
}

fn align(offset: u64) -> u64 {
    offset.next_multiple_of(SECTION_ALIGN as u64)
}

fn header_for(index: &HnswIndex, layout: &Layout, wal_seq: u64, body_crc32: u32) -> SnapshotHeader {
    let params = index.params();
    let selection = params.selection();
    SnapshotHeader {
        dim: index.dim() as u32,
        count: index.len() as u32,
        // Nothing is deleted yet, so every node is live. Phase 4 makes these differ.
        live_count: index.len() as u32,
        entry_point: index.entry_point(),
        rng_seed: params.seed(),
        levels_drawn: index.levels_drawn(),
        wal_seq,
        m: params.m() as u16,
        max_degree0: params.max_degree0() as u16,
        ef_construction: params.ef_construction() as u16,
        metric: index.metric(),
        max_layer: index.max_layer() as u8,
        section_offsets: layout.offsets,
        body_len: layout.body_len(),
        body_crc32,
        flags: HeaderFlags {
            heuristic: selection.heuristic,
            keep_pruned: selection.keep_pruned,
        },
    }
}

fn write_body<W: Write>(
    out: &mut CrcWriter<W>,
    index: &HnswIndex,
    layout: &Layout,
) -> std::io::Result<()> {
    // A fully in-memory index is one buffer and goes out in a single call; a hybrid one — a
    // snapshot with replayed vectors after it — has to be walked. Both produce the same bytes.
    match index.vectors().as_contiguous() {
        Some(data) => out.write_all(bytemuck::cast_slice(data))?,
        None => {
            let view = index.vectors().view();
            for position in 0..view.len() {
                out.write_all(bytemuck::cast_slice(view.get(position)))?;
            }
        }
    }
    // Each section is checked against the layout as it is written, so a size the header promises
    // and a size the body delivers cannot drift apart silently.
    debug_assert_eq!(out.written() as usize, layout.vectors);
    out.pad_to(layout.offsets[Section::NodeLevels as usize])?;

    let levels: Vec<u8> = (0..index.len())
        .map(|node| {
            index
                .level_of(node as NodeId)
                .expect("every node in the index has a level") as u8
        })
        .collect();
    debug_assert_eq!(levels.len(), layout.node_levels);
    out.write_all(&levels)?;
    out.pad_to(layout.offsets[Section::Layers as usize])?;

    let before = out.written();
    for (lc, layer) in index.layers().iter().enumerate() {
        if lc > 0 {
            out.write_all(&(layer.slot_nodes().len() as u32).to_le_bytes())?;
            out.write_all(bytemuck::cast_slice(layer.slot_nodes()))?;
        }
        out.write_all(bytemuck::cast_slice(layer.raw_neighbors()))?;
    }
    debug_assert_eq!((out.written() - before) as usize, layout.layers);
    out.pad_to(layout.body_len())?;

    Ok(())
}

fn params_from(header: &SnapshotHeader) -> Result<HnswParams, SnapshotError> {
    Ok(HnswParams::new(header.m as usize)
        .and_then(|p| p.with_max_degree0(header.max_degree0 as usize))
        .and_then(|p| p.with_ef_construction(header.ef_construction as usize))
        .and_then(|p| p.with_seed(header.rng_seed))
        .and_then(|p| {
            p.with_selection(SelectionPolicy {
                heuristic: header.flags.heuristic,
                keep_pruned: header.flags.keep_pruned,
            })
        })?)
}

/// Where `section`'s content starts within the body, given how long it must be.
///
/// Every section's length is computable from the header, so the reader never takes the file's word
/// for it — it works out what the section has to be and rejects anything else. The range a section
/// occupies is its content rounded up to [`SECTION_ALIGN`]; since section starts are aligned too
/// (the header enforces it), the expected range length is exactly `align(content)`, which makes
/// this an equality rather than a bound.
fn section_start(
    header: &SnapshotHeader,
    section: Section,
    content: usize,
) -> Result<usize, SnapshotError> {
    let range = header.section_range(section);
    let expected = (content as u64).next_multiple_of(SECTION_ALIGN as u64) as usize;
    if range.len() != expected {
        return Err(SnapshotError::SectionSizeMismatch {
            section: section.name(),
            found: range.len(),
            expected,
        });
    }
    Ok(range.start)
}

/// A section's content, with its padding trimmed off.
fn section<'a>(
    body: &'a [u8],
    header: &SnapshotHeader,
    which: Section,
    content: usize,
) -> Result<&'a [u8], SnapshotError> {
    let start = section_start(header, which, content)?;
    Ok(&body[start..start + content])
}

fn read_layers(
    body: &[u8],
    header: &SnapshotHeader,
    params: &HnswParams,
    count: usize,
) -> Result<Vec<Layer>, SnapshotError> {
    // The layers section is the one whose length is *not* derivable from the header alone — it
    // depends on how many nodes reached each layer. So it is read as far as it goes and then
    // required to have been consumed exactly, which catches the same disagreement from the other
    // side. Anything past the content is padding, at most `SECTION_ALIGN - 1` bytes.
    let range = header.section_range(Section::Layers);
    let mut cursor = Cursor::new(&body[range]);
    let mut layers = Vec::with_capacity(header.max_layer as usize + 1);

    for lc in 0..=header.max_layer as usize {
        let max_degree = params.max_degree(lc);
        let stride = max_degree + 1;

        let (nodes, slots) = if lc == 0 {
            // Layer 0 holds every node, so its length is known rather than stored.
            (Vec::new(), count)
        } else {
            let slots = cursor.u32()? as usize;
            // A corrupt count would otherwise ask for an allocation the size of the machine
            // before the length check below could reject it.
            if slots > count {
                return Err(SnapshotError::SectionSizeMismatch {
                    section: Section::Layers.name(),
                    found: slots,
                    expected: count,
                });
            }
            (cursor.u32_slice(slots)?.to_vec(), slots)
        };

        let neighbors = cursor.u32_slice(slots * stride)?.to_vec();
        layers.push(
            Layer::from_parts(max_degree, lc == 0, nodes, neighbors)
                .map_err(|e| SnapshotError::Index(anka_index::IndexError::LayerShape(e)))?,
        );
    }

    // Anything left beyond the alignment padding means the header and the body disagree about how
    // many layers there are, or how big they are.
    if cursor.remaining() >= SECTION_ALIGN {
        return Err(SnapshotError::SectionSizeMismatch {
            section: Section::Layers.name(),
            found: cursor.consumed() + cursor.remaining(),
            expected: cursor.consumed(),
        });
    }
    Ok(layers)
}

/// Sequential reader over the layers section.
///
/// Every read is length-checked and every cast goes through `try_cast_slice`, because the bytes
/// come from a file that may be anything at all. A misaligned cast would panic, and "corrupt file
/// takes down the process" is the failure this whole crate is written to avoid.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn consumed(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], SnapshotError> {
        if len > self.remaining() {
            return Err(SnapshotError::SectionSizeMismatch {
                section: Section::Layers.name(),
                found: self.remaining(),
                expected: len,
            });
        }
        let out = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, SnapshotError> {
        let bytes = self.take(size_of::<u32>())?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
    }

    fn u32_slice(&mut self, len: usize) -> Result<&'a [u32], SnapshotError> {
        let bytes = self.take(len * size_of::<u32>())?;
        bytemuck::try_cast_slice(bytes).map_err(|reason| SnapshotError::SectionNotCastable {
            section: Section::Layers.name(),
            element: "u32",
            reason,
        })
    }
}

/// Wraps a writer, checksumming everything on its way through and counting the bytes.
///
/// The body is hashed as it is written rather than read back afterwards: a second pass over
/// 700 MB to compute a checksum we could have accumulated for free is the kind of cost that
/// makes people turn checksums off.
struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
    written: u64,
}

impl<W: Write> CrcWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            written: 0,
        }
    }

    /// Bytes written so far. The writer is handed the body only, so this is a body offset.
    fn written(&self) -> u64 {
        self.written
    }

    /// Zero-fills up to `body_offset`, so the next section starts where the header says it does.
    fn pad_to(&mut self, body_offset: u64) -> std::io::Result<()> {
        let padding = body_offset - self.written();
        debug_assert!(padding < SECTION_ALIGN as u64);
        self.write_all(&[0u8; SECTION_ALIGN][..padding as usize])
    }

    fn finish(mut self) -> std::io::Result<u32> {
        self.inner.flush()?;
        Ok(self.hasher.finalize())
    }
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Makes the rename durable. See [`crate::fsync::parent_directory`] for why this step matters.
fn sync_parent_directory(path: &Path) -> Result<(), SnapshotError> {
    crate::fsync::parent_directory(path).map_err(|e| SnapshotError::io(path, e))
}

#[cfg(test)]
mod tests {
    use anka_core::{L2Squared, MetricKind};
    use anka_index::{DistanceCounter, HnswParams, SelectionPolicy};
    use tempfile::TempDir;

    use super::*;

    /// Deterministic points, so a failure is reproducible from its seed.
    fn points(seed: u64, count: usize, dim: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count * dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 100.0
            })
            .collect()
    }

    fn build(params: HnswParams, data: &[f32], dim: usize) -> HnswIndex {
        let mut index =
            HnswIndex::with_capacity(dim, MetricKind::L2Squared, params, data.len() / dim.max(1))
                .unwrap();
        let mut searcher = index.searcher();
        let mut counter = DistanceCounter::new();
        for vector in data.chunks_exact(dim) {
            index
                .insert::<L2Squared>(&mut searcher, vector, &mut counter)
                .unwrap();
        }
        index
    }

    /// Asserts the two graphs are the same edge for edge, in the same order.
    ///
    /// Deliberately not `graph_stats() == graph_stats()`: those byte counts come from `capacity()`
    /// and so measure the allocator's slack, not the graph. A layer grown by repeated pushes and
    /// one rebuilt with `to_vec` hold identical adjacency at different capacities, and the first
    /// version of this test failed on exactly that — a difference of nothing.
    fn assert_same_graph(a: &HnswIndex, b: &HnswIndex) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.max_layer(), b.max_layer());
        for layer in 0..=a.max_layer() {
            let (left, right) = (&a.layers()[layer], &b.layers()[layer]);
            assert_eq!(
                left.nodes().collect::<Vec<_>>(),
                right.nodes().collect::<Vec<_>>(),
                "layer {layer} holds different nodes"
            );
            for node in left.nodes() {
                assert_eq!(
                    left.neighbors(node),
                    right.neighbors(node),
                    "layer {layer}, node {node}"
                );
            }
        }
        for node in 0..a.len() as NodeId {
            assert_eq!(a.level_of(node), b.level_of(node), "level of node {node}");
        }
    }

    /// Asserts the two indexes answer every query identically — ids *and* distances, since
    /// `Candidate` compares on `(dist, id)`.
    fn assert_same_answers(a: &HnswIndex, b: &HnswIndex, queries: &[f32], dim: usize) {
        let mut sa = a.searcher();
        let mut sb = b.searcher();
        let mut counter = DistanceCounter::new();
        for query in queries.chunks_exact(dim) {
            let left = a
                .search::<L2Squared>(&mut sa, query, 10, 64, &mut counter)
                .unwrap();
            let right = b
                .search::<L2Squared>(&mut sb, query, 10, 64, &mut counter)
                .unwrap();
            assert_eq!(left, right);
        }
    }

    /// The phase 3 definition of done, at a size that runs in a unit test: write a real index,
    /// map it back, and get bit-identical answers.
    #[test]
    fn an_index_survives_a_round_trip_through_a_file() {
        let dim = 8;
        let data = points(1, 2_000, dim);
        let queries = points(2, 60, dim);
        let original = build(HnswParams::default(), &data, dim);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("collection.anka");
        write(&original, &path, 0).expect("write");

        let loaded = load(&path, Verify::Body).expect("load");
        loaded.validate().expect("a loaded graph is valid");

        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.dim(), original.dim());
        assert_eq!(loaded.metric(), original.metric());
        assert_eq!(loaded.max_layer(), original.max_layer());
        assert_eq!(loaded.entry_point(), original.entry_point());
        assert_eq!(loaded.levels_drawn(), original.levels_drawn());
        assert_eq!(loaded.params(), original.params());
        assert_same_graph(&original, &loaded);
        assert_same_answers(&original, &loaded, &queries, dim);
    }

    /// The point of mapping rather than reading: the vectors are never copied into the process.
    #[test]
    fn loaded_vectors_are_mapped_not_copied() {
        let dim = 4;
        let data = points(3, 500, dim);
        let index = build(HnswParams::default(), &data, dim);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&index, &path, 0).unwrap();
        let loaded = load(&path, Verify::Header).unwrap();

        assert!(loaded.vectors().is_mapped());
        assert_eq!(loaded.vectors().mapped_count(), data.len() / dim);
        assert_eq!(loaded.vectors().resident_bytes(), 0);
        // Same bytes, read through the mapping.
        for position in 0..loaded.len() {
            assert_eq!(
                loaded.vectors().get(position),
                &data[position * dim..(position + 1) * dim]
            );
        }
    }

    /// The two loaders differ only in where the vectors end up. Everything they produce — the
    /// graph, the answers, the distances — has to be identical, or the comparison `anka snapshot`
    /// reports would be measuring two different indexes.
    #[test]
    fn reading_and_mapping_produce_the_same_index() {
        let dim = 8;
        let data = points(17, 1_000, dim);
        let queries = points(18, 40, dim);
        let original = build(HnswParams::default(), &data, dim);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&original, &path, 0).unwrap();

        let mapped = load(&path, Verify::Body).unwrap();
        let owned = read(&path, Verify::Body).unwrap();

        assert!(mapped.vectors().is_mapped());
        assert!(!owned.vectors().is_mapped());
        assert_eq!(owned.vectors().resident_bytes(), data.len() * 4);

        assert_same_graph(&mapped, &owned);
        assert_same_answers(&mapped, &owned, &queries, dim);
        assert_same_answers(&original, &owned, &queries, dim);
    }

    /// Reading is the same parser, so it has to reject the same files.
    #[test]
    fn reading_rejects_what_mapping_rejects() {
        let dim = 4;
        let index = build(HnswParams::default(), &points(19, 100, dim), dim);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&index, &path, 0).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_BYTES + 8] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        assert!(matches!(
            read(&path, Verify::Body),
            Err(SnapshotError::BodyChecksumMismatch { .. })
        ));
    }

    /// Every parameter that changes the graph's shape has to come back, or a reloaded index would
    /// keep building differently from the one that was saved.
    #[test]
    fn build_parameters_survive_the_round_trip() {
        let dim = 4;
        let data = points(4, 300, dim);
        let params = HnswParams::new(6)
            .unwrap()
            .with_max_degree0(20)
            .unwrap()
            .with_ef_construction(48)
            .unwrap()
            .with_seed(9_876_543_210)
            .unwrap()
            .with_selection(SelectionPolicy {
                heuristic: true,
                keep_pruned: false,
            })
            .unwrap();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&build(params, &data, dim), &path, 0).unwrap();

        assert_eq!(load(&path, Verify::Body).unwrap().params(), &params);
    }

    /// A checkpoint is not the end of an index's life: it keeps taking inserts afterwards, and
    /// they have to land on the same levels they would have without the restart.
    #[test]
    fn a_loaded_index_keeps_inserting_where_it_left_off() {
        let dim = 4;
        let data = points(5, 400, dim);
        let extra = points(6, 40, dim);
        let mut original = build(HnswParams::default(), &data, dim);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&original, &path, 0).unwrap();
        let mut loaded = load(&path, Verify::Body).unwrap();

        for index in [&mut original, &mut loaded] {
            let mut searcher = index.searcher();
            let mut counter = DistanceCounter::new();
            for vector in extra.chunks_exact(dim) {
                index
                    .insert::<L2Squared>(&mut searcher, vector, &mut counter)
                    .unwrap();
            }
        }

        // The reloaded store was mapped and became hybrid on the first push — the transition
        // phase 3b exists for.
        assert!(loaded.vectors().is_mapped());
        assert_eq!(loaded.vectors().mapped_count(), data.len() / dim);

        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.levels_drawn(), original.levels_drawn());
        assert_same_graph(&original, &loaded);
        assert_same_answers(&original, &loaded, &points(7, 40, dim), dim);
    }

    /// A hybrid store has its vectors in two places; writing it has to produce the same file as
    /// writing an index that was built in one go.
    #[test]
    fn a_hybrid_store_writes_the_same_bytes_as_an_owned_one() {
        let dim = 4;
        let data = points(8, 300, dim);
        let extra = points(9, 30, dim);

        let dir = TempDir::new().unwrap();
        let first = dir.path().join("first.anka");
        write(&build(HnswParams::default(), &data, dim), &first, 0).unwrap();

        let mut reloaded = load(&first, Verify::Body).unwrap();
        let mut searcher = reloaded.searcher();
        let mut counter = DistanceCounter::new();
        for vector in extra.chunks_exact(dim) {
            reloaded
                .insert::<L2Squared>(&mut searcher, vector, &mut counter)
                .unwrap();
        }
        assert!(reloaded.vectors().as_contiguous().is_none());

        let mut all = data.clone();
        all.extend_from_slice(&extra);
        let built_in_one_go = build(HnswParams::default(), &all, dim);

        let hybrid_path = dir.path().join("hybrid.anka");
        let owned_path = dir.path().join("owned.anka");
        write(&reloaded, &hybrid_path, 0).unwrap();
        write(&built_in_one_go, &owned_path, 0).unwrap();

        assert_eq!(
            std::fs::read(&hybrid_path).unwrap(),
            std::fs::read(&owned_path).unwrap()
        );
    }

    #[test]
    fn an_empty_index_round_trips() {
        let index = HnswIndex::new(6, MetricKind::L2Squared, HnswParams::default()).unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.anka");
        write(&index, &path, 0).unwrap();

        let loaded = load(&path, Verify::Body).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.dim(), 6);
        assert_eq!(loaded.entry_point(), None);
        loaded.validate().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), HEADER_BYTES as u64);
    }

    #[test]
    fn the_wal_sequence_is_carried_through() {
        let dim = 4;
        let index = build(HnswParams::default(), &points(10, 50, dim), dim);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&index, &path, 4_242).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(SnapshotHeader::decode(&bytes).unwrap().wal_seq, 4_242);
    }

    /// The temporary file must not survive a successful write, or the next one would be writing
    /// over a name that already exists and a `.tmp` would sit next to every collection.
    #[test]
    fn the_temporary_file_is_gone_afterwards() {
        let dim = 4;
        let index = build(HnswParams::default(), &points(11, 50, dim), dim);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&index, &path, 0).unwrap();

        assert!(path.exists());
        assert!(!dir.path().join("c.anka.tmp").exists());
    }

    /// Overwriting is the checkpoint path. The new snapshot has to replace the old one entirely,
    /// including when it is shorter — a leftover tail would checksum as corruption.
    #[test]
    fn overwriting_replaces_the_previous_snapshot_entirely() {
        let dim = 4;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");

        write(
            &build(HnswParams::default(), &points(12, 800, dim), dim),
            &path,
            0,
        )
        .unwrap();
        let large = std::fs::metadata(&path).unwrap().len();

        let small = build(HnswParams::default(), &points(13, 40, dim), dim);
        write(&small, &path, 0).unwrap();

        assert!(std::fs::metadata(&path).unwrap().len() < large);
        let loaded = load(&path, Verify::Body).unwrap();
        assert_eq!(loaded.len(), 40);
        loaded.validate().unwrap();
    }

    /// A damaged body has to be caught when asked for, and *not* scanned for when it is not:
    /// the whole point of two checksums is two policies.
    #[test]
    fn a_corrupt_body_is_caught_only_when_verification_is_asked_for() {
        let dim = 4;
        let index = build(HnswParams::default(), &points(14, 200, dim), dim);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&index, &path, 0).unwrap();

        let mut bytes = std::fs::read(&path).unwrap();
        let victim = HEADER_BYTES + 32;
        bytes[victim] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        assert!(matches!(
            load(&path, Verify::Body),
            Err(SnapshotError::BodyChecksumMismatch { .. })
        ));
        // The header is intact, so a header-only load still succeeds. That is the documented
        // trade, not an oversight: verifying 700 MB on every open defeats the mapping.
        assert!(load(&path, Verify::Header).is_ok());
    }

    /// Corruption anywhere in the file is an error, never a panic. The bytes come from disk and
    /// spec section 7 requires that a damaged snapshot cannot take the process down.
    #[test]
    fn corruption_is_rejected_rather_than_crashing() {
        let dim = 4;
        let index = build(HnswParams::default(), &points(15, 100, dim), dim);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&index, &path, 0).unwrap();
        let good = std::fs::read(&path).unwrap();

        // Truncated, at every length that still has a header.
        for len in [0, 1, HEADER_BYTES - 1, HEADER_BYTES, HEADER_BYTES + 7] {
            std::fs::write(&path, &good[..len.min(good.len())]).unwrap();
            assert!(
                load(&path, Verify::Body).is_err(),
                "a {len}-byte file was accepted"
            );
        }

        // A single flipped bit anywhere in the header, which the header checksum must catch.
        for byte in 0..HEADER_BYTES {
            let mut bytes = good.clone();
            bytes[byte] ^= 0x01;
            std::fs::write(&path, &bytes).unwrap();
            assert!(
                load(&path, Verify::Header).is_err(),
                "a flipped bit at header byte {byte} was accepted"
            );
        }

        // Foreign file.
        std::fs::write(&path, b"this is not a snapshot, it is a text file").unwrap();
        assert!(matches!(
            load(&path, Verify::Header),
            Err(SnapshotError::TooShort { .. } | SnapshotError::BadMagic { .. })
        ));
    }

    /// The header can be internally consistent and still describe a body that is not there. Every
    /// one of these has to be an error before the numbers are used to slice the mapping.
    #[test]
    fn a_header_that_disagrees_with_its_body_is_rejected() {
        let dim = 4;
        let index = build(HnswParams::default(), &points(16, 200, dim), dim);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.anka");
        write(&index, &path, 0).unwrap();
        let good = std::fs::read(&path).unwrap();

        let rewrite = |mutate: &dyn Fn(&mut SnapshotHeader)| {
            let mut header = SnapshotHeader::decode(&good).unwrap();
            mutate(&mut header);
            let mut bytes = good.clone();
            bytes[..HEADER_BYTES].copy_from_slice(&header.encode());
            std::fs::write(&path, &bytes).unwrap();
        };

        // Body longer than the file.
        rewrite(&|h| h.body_len += SECTION_ALIGN as u64);
        assert!(matches!(
            load(&path, Verify::Header),
            Err(SnapshotError::BodyTruncated { .. })
        ));

        // A dimension the vectors section cannot hold.
        rewrite(&|h| h.dim *= 2);
        assert!(load(&path, Verify::Header).is_err());

        // More vectors than there are bytes for.
        rewrite(&|h| h.count += 1);
        assert!(load(&path, Verify::Header).is_err());

        // Sections that overlap: node levels claimed to start where the vectors do.
        rewrite(&|h| h.section_offsets[Section::NodeLevels as usize] = 0);
        assert!(matches!(
            load(&path, Verify::Header),
            Err(SnapshotError::SectionSizeMismatch { .. })
        ));
    }
}
