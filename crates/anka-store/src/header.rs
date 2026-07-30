//! The snapshot header.
//!
//! 128 bytes, little-endian, written field by field rather than by casting a `#[repr(C)]` struct.
//! A struct cast would tie the on-disk layout to whatever padding the compiler chooses, and a file
//! format has to survive a compiler upgrade.
//!
//! ```text
//! off  size  field
//!   0     4  magic "ANKA"
//!   4     4  format_version
//!   8     4  dim
//!  12     4  count            NodeId space, live + tombstoned
//!  16     4  live_count
//!  20     4  entry_point      u32::MAX means none
//!  24     8  rng_seed
//!  32     8  levels_drawn     how many levels the generator has produced
//!  40     8  wal_seq          this snapshot contains WAL records up to and including this
//!  48     2  M
//!  50     2  max_degree0
//!  52     2  ef_construction
//!  54     1  metric
//!  55     1  max_layer
//!  56    48  section_offsets[6]
//! 104     8  body_len
//! 112     4  body_crc32
//! 116     4  flags            how the graph was built
//! 120     4  reserved
//! 124     4  header_crc32     covers bytes 0..124
//! ```
//!
//! **The header checksum covers every preceding byte, including the section offsets.** The spec
//! originally scoped it to the first 96 bytes, which would have left the offsets unprotected: a
//! corrupted offset would pass the header check and then be used to slice into the mapping. A
//! checksum at the end covering everything before it is both conventional and unambiguous.
//!
//! Two checksums with two policies, which is the point of having two. `header_crc32` is verified on
//! every open — 124 bytes, free, and it catches a truncated or foreign file immediately.
//! `body_crc32` is verified only on request, in crash tests and in CI: scanning 700 MB on every
//! open would defeat the lazy, zero-copy mapping the format exists to enable.

use anka_core::{MAX_DIM, MetricKind, NodeId};

use crate::error::SnapshotError;

pub const MAGIC: [u8; 4] = *b"ANKA";

/// On-disk format version.
///
/// Starts at 1. The spec numbers this 2 because the *spec* revised the layout between v1.0 and
/// v1.1, but no version ever reached a disk, so there is nothing to be compatible with.
pub const FORMAT_VERSION: u32 = 1;

pub const HEADER_BYTES: usize = 128;

/// Bytes covered by `header_crc32`: everything before it.
const HEADER_CHECKSUM_BYTES: usize = 124;

pub const SECTION_COUNT: usize = 6;

/// Sections of the snapshot body, in the order they are laid out.
///
/// The order is fixed because the offsets are validated as non-decreasing, which is what turns a
/// corrupt offset table into an error rather than an out-of-bounds read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Vectors = 0,
    NodeLevels = 1,
    Layers = 2,
    Tombstones = 3,
    IdMap = 4,
    Metadata = 5,
}

impl Section {
    pub const ALL: [Section; SECTION_COUNT] = [
        Section::Vectors,
        Section::NodeLevels,
        Section::Layers,
        Section::Tombstones,
        Section::IdMap,
        Section::Metadata,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Vectors => "vectors",
            Self::NodeLevels => "node_levels",
            Self::Layers => "layers",
            Self::Tombstones => "tombstones",
            Self::IdMap => "id_map",
            Self::Metadata => "metadata",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// How the graph was built. Recorded because the graph's shape depends on it, so a snapshot that
/// did not carry it could not be interpreted — or reproduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeaderFlags {
    pub heuristic: bool,
    pub keep_pruned: bool,
}

impl HeaderFlags {
    const HEURISTIC: u32 = 1 << 0;
    const KEEP_PRUNED: u32 = 1 << 1;
    const KNOWN: u32 = Self::HEURISTIC | Self::KEEP_PRUNED;

    fn to_bits(self) -> u32 {
        let mut bits = 0;
        if self.heuristic {
            bits |= Self::HEURISTIC;
        }
        if self.keep_pruned {
            bits |= Self::KEEP_PRUNED;
        }
        bits
    }

    /// Unknown bits are an error, not something to ignore.
    ///
    /// A future version setting a flag we do not understand has, by definition, built a graph whose
    /// shape we cannot reason about. Silently dropping the bit would produce plausible, wrong
    /// results — the failure mode this project spends most of its effort avoiding.
    fn from_bits(bits: u32) -> Result<Self, SnapshotError> {
        if bits & !Self::KNOWN != 0 {
            return Err(SnapshotError::UnknownFlags {
                bits: bits & !Self::KNOWN,
            });
        }
        Ok(Self {
            heuristic: bits & Self::HEURISTIC != 0,
            keep_pruned: bits & Self::KEEP_PRUNED != 0,
        })
    }
}

/// Everything needed to interpret a snapshot body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotHeader {
    pub dim: u32,
    pub count: u32,
    pub live_count: u32,
    pub entry_point: Option<NodeId>,
    pub rng_seed: u64,
    pub levels_drawn: u64,
    pub wal_seq: u64,
    pub m: u16,
    pub max_degree0: u16,
    pub ef_construction: u16,
    pub metric: MetricKind,
    pub max_layer: u8,
    pub section_offsets: [u64; SECTION_COUNT],
    pub body_len: u64,
    pub body_crc32: u32,
    pub flags: HeaderFlags,
}

/// `entry_point` sentinel for "none".
///
/// Safe: `count <= u32::MAX` and a valid entry point is `< count`, so `u32::MAX` can never be one.
const NO_ENTRY_POINT: u32 = u32::MAX;

impl SnapshotHeader {
    /// Serialises to exactly [`HEADER_BYTES`] bytes, checksum included.
    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut buf = [0u8; HEADER_BYTES];
        let mut w = Writer::new(&mut buf);

        w.bytes(&MAGIC);
        w.u32(FORMAT_VERSION);
        w.u32(self.dim);
        w.u32(self.count);
        w.u32(self.live_count);
        w.u32(self.entry_point.unwrap_or(NO_ENTRY_POINT));
        w.u64(self.rng_seed);
        w.u64(self.levels_drawn);
        w.u64(self.wal_seq);
        w.u16(self.m);
        w.u16(self.max_degree0);
        w.u16(self.ef_construction);
        w.u8(self.metric.as_u8());
        w.u8(self.max_layer);
        for offset in self.section_offsets {
            w.u64(offset);
        }
        w.u64(self.body_len);
        w.u32(self.body_crc32);
        w.u32(self.flags.to_bits());
        w.u32(0); // reserved

        debug_assert_eq!(w.position(), HEADER_CHECKSUM_BYTES);
        let checksum = crc32(&buf[..HEADER_CHECKSUM_BYTES]);
        buf[HEADER_CHECKSUM_BYTES..].copy_from_slice(&checksum.to_le_bytes());
        buf
    }

    /// Parses and validates a header.
    ///
    /// Every failure is an error rather than a panic: a snapshot is an untrusted file, and the
    /// fuzz requirement in spec section 7 is exactly that a corrupt one must not bring the process
    /// down.
    pub fn decode(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.len() < HEADER_BYTES {
            return Err(SnapshotError::TooShort {
                needed: HEADER_BYTES,
                got: bytes.len(),
            });
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[..4]);
        if magic != MAGIC {
            return Err(SnapshotError::BadMagic { found: magic });
        }

        // Checksum before interpreting anything else, so no field is trusted before the bytes are.
        let stored = u32::from_le_bytes(
            bytes[HEADER_CHECKSUM_BYTES..HEADER_BYTES]
                .try_into()
                .expect("4 bytes"),
        );
        let computed = crc32(&bytes[..HEADER_CHECKSUM_BYTES]);
        if stored != computed {
            return Err(SnapshotError::HeaderChecksumMismatch { stored, computed });
        }

        let mut r = Reader::new(bytes);
        r.skip(4); // magic, already checked

        let version = r.u32();
        if version != FORMAT_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }

        let dim = r.u32();
        let count = r.u32();
        let live_count = r.u32();
        let entry_point_raw = r.u32();
        let rng_seed = r.u64();
        let levels_drawn = r.u64();
        let wal_seq = r.u64();
        let m = r.u16();
        let max_degree0 = r.u16();
        let ef_construction = r.u16();
        let metric_tag = r.u8();
        let max_layer = r.u8();

        let mut section_offsets = [0u64; SECTION_COUNT];
        for offset in &mut section_offsets {
            *offset = r.u64();
        }
        let body_len = r.u64();
        let body_crc32 = r.u32();
        let flags = HeaderFlags::from_bits(r.u32())?;

        let header = Self {
            dim,
            count,
            live_count,
            entry_point: (entry_point_raw != NO_ENTRY_POINT).then_some(entry_point_raw),
            rng_seed,
            levels_drawn,
            wal_seq,
            m,
            max_degree0,
            ef_construction,
            metric: MetricKind::from_u8(metric_tag)
                .ok_or(SnapshotError::UnknownMetric { tag: metric_tag })?,
            max_layer,
            section_offsets,
            body_len,
            body_crc32,
            flags,
        };
        header.validate()?;
        Ok(header)
    }

    /// Consistency checks that the checksum cannot catch.
    ///
    /// A file can be bit-perfect and still describe something impossible — a zero dimension, an
    /// entry point outside the collection, sections that overlap. Those have to be rejected before
    /// any of the numbers are used to slice a mapping.
    fn validate(&self) -> Result<(), SnapshotError> {
        if self.dim == 0 || self.dim as usize > MAX_DIM {
            return Err(SnapshotError::InvalidDim { dim: self.dim });
        }
        if self.live_count > self.count {
            return Err(SnapshotError::LiveCountAboveCount {
                live_count: self.live_count,
                count: self.count,
            });
        }
        if let Some(entry) = self.entry_point
            && entry >= self.count
        {
            return Err(SnapshotError::EntryPointOutOfRange {
                entry_point: entry,
                count: self.count,
            });
        }
        if self.entry_point.is_none() && self.count > 0 {
            return Err(SnapshotError::MissingEntryPoint { count: self.count });
        }
        if self.m == 0 || self.max_degree0 < self.m {
            return Err(SnapshotError::InvalidDegrees {
                m: self.m,
                max_degree0: self.max_degree0,
            });
        }
        if self.ef_construction == 0 {
            return Err(SnapshotError::InvalidEfConstruction);
        }

        // Offsets are non-decreasing and inside the body. Together these two checks make every
        // section a well-formed range, so the reader can slice without further arithmetic.
        let mut previous = 0u64;
        for section in Section::ALL {
            let offset = self.section_offsets[section.index()];
            if offset < previous {
                return Err(SnapshotError::SectionsOutOfOrder {
                    section: section.name(),
                    offset,
                    previous,
                });
            }
            if offset > self.body_len {
                return Err(SnapshotError::SectionOutOfRange {
                    section: section.name(),
                    offset,
                    body_len: self.body_len,
                });
            }
            previous = offset;
        }
        Ok(())
    }

    /// Byte range of `section` within the body.
    ///
    /// The last section runs to `body_len`; every other one ends where the next begins. Valid by
    /// construction, since [`Self::validate`] has already established the offsets are ordered and
    /// bounded.
    pub fn section_range(&self, section: Section) -> std::ops::Range<usize> {
        let start = self.section_offsets[section.index()];
        let end = Section::ALL
            .get(section.index() + 1)
            .map_or(self.body_len, |next| self.section_offsets[next.index()]);
        start as usize..end as usize
    }

    /// Where the body starts in the file.
    pub const fn body_start() -> usize {
        HEADER_BYTES
    }
}

pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// Sequential little-endian field writer.
///
/// Deliberately mirrored by [`Reader`] method for method: a format whose write and read sides are
/// spelled differently is a format where a field eventually gets read at the wrong offset.
struct Writer<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> Writer<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn bytes(&mut self, value: &[u8]) {
        self.buf[self.pos..self.pos + value.len()].copy_from_slice(value);
        self.pos += value.len();
    }

    fn u8(&mut self, value: u8) {
        self.bytes(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }
}

/// Sequential little-endian field reader.
///
/// Callers must have length-checked the buffer first; within this module that is guaranteed by the
/// `HEADER_BYTES` check at the top of `decode`.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        out.copy_from_slice(&self.buf[self.pos..self.pos + N]);
        self.pos += N;
        out
    }

    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }

    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take())
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SnapshotHeader {
        SnapshotHeader {
            dim: 128,
            count: 1_000_000,
            live_count: 999_999,
            entry_point: Some(42),
            rng_seed: 0xDEAD_BEEF_CAFE_F00D,
            levels_drawn: 1_000_000,
            wal_seq: 7,
            m: 16,
            max_degree0: 32,
            ef_construction: 200,
            metric: MetricKind::L2Squared,
            max_layer: 4,
            section_offsets: [
                0,
                512_000_000,
                512_001_000,
                700_000_000,
                700_000_100,
                700_000_200,
            ],
            body_len: 700_000_300,
            body_crc32: 0x1234_5678,
            flags: HeaderFlags {
                heuristic: true,
                keep_pruned: true,
            },
        }
    }

    #[test]
    fn the_header_is_exactly_128_bytes() {
        assert_eq!(sample().encode().len(), HEADER_BYTES);
        assert_eq!(HEADER_BYTES, 128);
    }

    #[test]
    fn round_trips() {
        let header = sample();
        assert_eq!(SnapshotHeader::decode(&header.encode()).unwrap(), header);
    }

    #[test]
    fn an_empty_collection_round_trips() {
        let header = SnapshotHeader {
            count: 0,
            live_count: 0,
            entry_point: None,
            max_layer: 0,
            levels_drawn: 0,
            section_offsets: [0; SECTION_COUNT],
            body_len: 0,
            ..sample()
        };
        let decoded = SnapshotHeader::decode(&header.encode()).unwrap();
        assert_eq!(decoded, header);
        assert_eq!(decoded.entry_point, None);
    }

    /// Field offsets are part of the format, so they are asserted rather than left to whatever
    /// the writer happens to do.
    #[test]
    fn key_fields_sit_at_their_documented_offsets() {
        let bytes = sample().encode();
        assert_eq!(&bytes[0..4], b"ANKA");
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            FORMAT_VERSION
        );
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 128);
        assert_eq!(
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_F00D
        );
        // Checksum is the last four bytes.
        assert_eq!(
            u32::from_le_bytes(bytes[124..128].try_into().unwrap()),
            crc32(&bytes[..124])
        );
    }

    /// The claim this format makes about its own checksum, tested directly: flipping *any* byte
    /// before the checksum must be caught. This is what protects the section offsets, which the
    /// spec's original 96-byte scope would have left exposed.
    #[test]
    fn every_byte_before_the_checksum_is_protected() {
        let original = sample().encode();
        for index in 0..HEADER_CHECKSUM_BYTES {
            let mut corrupted = original;
            corrupted[index] ^= 0x01;
            let result = SnapshotHeader::decode(&corrupted);
            assert!(
                result.is_err(),
                "flipping byte {index} produced a header that decoded cleanly"
            );
        }
    }

    #[test]
    fn a_short_buffer_is_rejected() {
        let bytes = sample().encode();
        for length in [0usize, 1, 4, 64, 127] {
            assert!(matches!(
                SnapshotHeader::decode(&bytes[..length]),
                Err(SnapshotError::TooShort { .. })
            ));
        }
    }

    #[test]
    fn a_foreign_file_is_rejected_before_anything_else() {
        let mut bytes = sample().encode();
        bytes[..4].copy_from_slice(b"XXXX");
        assert!(matches!(
            SnapshotHeader::decode(&bytes),
            Err(SnapshotError::BadMagic { .. })
        ));
    }

    #[test]
    fn an_unsupported_version_is_rejected() {
        let mut header = sample().encode();
        header[4..8].copy_from_slice(&99u32.to_le_bytes());
        // Re-checksum, so the version check is what fires rather than the CRC.
        let checksum = crc32(&header[..HEADER_CHECKSUM_BYTES]);
        header[HEADER_CHECKSUM_BYTES..].copy_from_slice(&checksum.to_le_bytes());

        assert!(matches!(
            SnapshotHeader::decode(&header),
            Err(SnapshotError::UnsupportedVersion {
                found: 99,
                supported: FORMAT_VERSION
            })
        ));
    }

    /// Re-checksums after editing, so each test exercises the validation rule it names rather than
    /// the CRC.
    fn encode_with(mutate: impl FnOnce(&mut SnapshotHeader)) -> [u8; HEADER_BYTES] {
        let mut header = sample();
        mutate(&mut header);
        header.encode()
    }

    #[test]
    fn a_zero_or_absurd_dimension_is_rejected() {
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.dim = 0)),
            Err(SnapshotError::InvalidDim { dim: 0 })
        ));
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.dim = MAX_DIM as u32 + 1)),
            Err(SnapshotError::InvalidDim { .. })
        ));
    }

    #[test]
    fn an_entry_point_outside_the_collection_is_rejected() {
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.entry_point = Some(2_000_000))),
            Err(SnapshotError::EntryPointOutOfRange { .. })
        ));
    }

    /// A non-empty collection with no entry point cannot be searched at all.
    #[test]
    fn a_populated_snapshot_without_an_entry_point_is_rejected() {
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.entry_point = None)),
            Err(SnapshotError::MissingEntryPoint { .. })
        ));
    }

    #[test]
    fn more_live_vectors_than_total_is_rejected() {
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.live_count = h.count + 1)),
            Err(SnapshotError::LiveCountAboveCount { .. })
        ));
    }

    #[test]
    fn degenerate_degree_parameters_are_rejected() {
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.m = 0)),
            Err(SnapshotError::InvalidDegrees { .. })
        ));
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.max_degree0 = h.m - 1)),
            Err(SnapshotError::InvalidDegrees { .. })
        ));
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.ef_construction = 0)),
            Err(SnapshotError::InvalidEfConstruction)
        ));
    }

    /// Out-of-order or out-of-bounds offsets have to be caught here, because everything downstream
    /// uses them to slice a memory mapping.
    #[test]
    fn malformed_section_offsets_are_rejected() {
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.section_offsets[2] = 1)),
            Err(SnapshotError::SectionsOutOfOrder { .. })
        ));
        assert!(matches!(
            SnapshotHeader::decode(&encode_with(|h| h.section_offsets[5] = h.body_len + 1)),
            Err(SnapshotError::SectionOutOfRange { .. })
        ));
    }

    #[test]
    fn an_unknown_metric_tag_is_rejected() {
        let mut bytes = sample().encode();
        bytes[54] = 200;
        let checksum = crc32(&bytes[..HEADER_CHECKSUM_BYTES]);
        bytes[HEADER_CHECKSUM_BYTES..].copy_from_slice(&checksum.to_le_bytes());

        assert!(matches!(
            SnapshotHeader::decode(&bytes),
            Err(SnapshotError::UnknownMetric { tag: 200 })
        ));
    }

    /// A flag we do not understand means a graph whose shape we cannot reason about. Dropping the
    /// bit would produce plausible, wrong results.
    #[test]
    fn unknown_flag_bits_are_rejected_rather_than_ignored() {
        let mut bytes = sample().encode();
        bytes[116..120].copy_from_slice(&(1u32 << 7).to_le_bytes());
        let checksum = crc32(&bytes[..HEADER_CHECKSUM_BYTES]);
        bytes[HEADER_CHECKSUM_BYTES..].copy_from_slice(&checksum.to_le_bytes());

        assert!(matches!(
            SnapshotHeader::decode(&bytes),
            Err(SnapshotError::UnknownFlags { .. })
        ));
    }

    #[test]
    fn flags_round_trip_in_every_combination() {
        for heuristic in [false, true] {
            for keep_pruned in [false, true] {
                let flags = HeaderFlags {
                    heuristic,
                    keep_pruned,
                };
                let bytes = encode_with(|h| h.flags = flags);
                assert_eq!(SnapshotHeader::decode(&bytes).unwrap().flags, flags);
            }
        }
    }

    #[test]
    fn section_ranges_tile_the_body_without_gaps() {
        let header = sample();
        let mut previous_end = 0usize;
        for section in Section::ALL {
            let range = header.section_range(section);
            assert_eq!(range.start, previous_end, "gap before {}", section.name());
            assert!(range.end >= range.start);
            previous_end = range.end;
        }
        assert_eq!(previous_end, header.body_len as usize);
    }

    #[test]
    fn the_body_starts_after_the_header() {
        assert_eq!(SnapshotHeader::body_start(), HEADER_BYTES);
    }
}
