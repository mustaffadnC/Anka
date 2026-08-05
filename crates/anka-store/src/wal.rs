//! The write-ahead log: format, encoding, and the appending writer.
//!
//! ```text
//! [ header 16B ]  magic "AWAL" | format_version u32 | reserved 8B
//!
//! then a stream of records:
//! [ len u32 ][ crc32 u32 ][ seq u64 ][ op u8 ][ payload ]
//!   len   = byte count of (seq ++ op ++ payload), excluding len and crc32
//!   crc32 = crc32(seq ++ op ++ payload), excluding len
//!
//! 0x01 INSERT      ext_id u64 + level u8 + dim × f32 + meta_len u32 + meta
//! 0x02 DELETE      ext_id u64
//! 0x03 CHECKPOINT  snapshot_wal_seq u64
//! ```
//!
//! Two fields here are easy to leave out and painful to add afterwards.
//!
//! **`seq`.** Recovery means "replay the records after the snapshot". Without a per-record
//! sequence number there is no definition of *after*, and recovery cannot be written at all.
//!
//! **`level`.** Layer assignment is random. If replay draws a fresh level, the recovered graph is
//! not the graph that was lost, and "identical results after a restart" stops being testable.
//!
//! **The vector's length is not stored**, because `dim` is fixed per collection and comes from the
//! snapshot. That is not a leap of faith: `len` and the payload arithmetic have to agree exactly,
//! so a record written at a different `dim` is a hard error rather than a misparse.
//!
//! **Torn tails versus wrong logs.** A record can fail to read for two very different reasons, and
//! collapsing them would either discard committed data or accept damaged data:
//!
//! - the log simply *stops* partway through a record — the process died mid-write. That is
//!   expected, handled, and the tail is truncated. [`Next::Torn`].
//! - the bytes are intact and their checksum is valid, but they say something this build does not
//!   understand — an unknown op, or a payload whose parts do not add up. Nothing was torn; a
//!   writer wrote something else. Dropping it would silently lose committed records, so it is an
//!   error. [`WalError`].

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use anka_core::{ExternalId, MAX_DIM};

use crate::error::WalError;
use crate::header::crc32;

pub const MAGIC: [u8; 4] = *b"AWAL";
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 16;

/// `len` and `crc32`, which sit outside what `len` counts.
pub const FRAME_BYTES: usize = 8;
/// What `len` counts before the payload: `seq` and `op`.
const PREFIX_BYTES: usize = 9;

/// Sanity bound on one record.
///
/// The widest supported vector is [`MAX_DIM`] × 4 bytes = 16 KiB, so this leaves ample room for
/// metadata while turning a corrupt `len` into an error instead of a scan across the whole file.
pub const MAX_RECORD_BYTES: usize = 16 << 20;

const OP_INSERT: u8 = 0x01;
const OP_DELETE: u8 = 0x02;
const OP_CHECKPOINT: u8 = 0x03;

/// One logged operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    Insert {
        external_id: ExternalId,
        /// The level the writer assigned. Replay uses it rather than drawing its own.
        level: u8,
        vector: Vec<f32>,
        /// Opaque until phase 4 gives it a schema; carried so the format does not change then.
        metadata: Vec<u8>,
    },
    Delete {
        external_id: ExternalId,
    },
    /// Marks that a snapshot containing everything up to `snapshot_wal_seq` is on disk.
    Checkpoint {
        snapshot_wal_seq: u64,
    },
}

impl Record {
    fn op(&self) -> u8 {
        match self {
            Self::Insert { .. } => OP_INSERT,
            Self::Delete { .. } => OP_DELETE,
            Self::Checkpoint { .. } => OP_CHECKPOINT,
        }
    }

    fn payload_len(&self) -> usize {
        match self {
            Self::Insert {
                vector, metadata, ..
            } => size_of::<u64>() + 1 + vector.len() * size_of::<f32>() + 4 + metadata.len(),
            Self::Delete { .. } => size_of::<u64>(),
            Self::Checkpoint { .. } => size_of::<u64>(),
        }
    }

    fn write_payload(&self, out: &mut Vec<u8>) {
        match self {
            Self::Insert {
                external_id,
                level,
                vector,
                metadata,
            } => {
                out.extend_from_slice(&external_id.to_le_bytes());
                out.push(*level);
                for value in vector {
                    out.extend_from_slice(&value.to_le_bytes());
                }
                out.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
                out.extend_from_slice(metadata);
            }
            Self::Delete { external_id } => out.extend_from_slice(&external_id.to_le_bytes()),
            Self::Checkpoint { snapshot_wal_seq } => {
                out.extend_from_slice(&snapshot_wal_seq.to_le_bytes())
            }
        }
    }

    /// Bytes this record occupies on disk, framing included.
    pub fn encoded_len(&self) -> usize {
        FRAME_BYTES + PREFIX_BYTES + self.payload_len()
    }
}

/// When the log is forced to disk.
///
/// The distinction that matters is *what a crash costs*, and the answer differs by crash. See
/// `docs/DESIGN.md`, section 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncPolicy {
    /// `fsync` after every record. Survives a power cut; the default, and the only mode under
    /// which this project claims no acknowledged record is lost.
    #[default]
    Always,
    /// `fsync` every N records. A power cut loses at most the last N.
    EveryN(NonZeroU32),
    /// Never `fsync`. A process crash loses nothing — the bytes are already in the page cache —
    /// but a power cut loses whatever the OS had not flushed.
    Never,
}

/// Appends records to a log file.
///
/// **Order of operations, and it is the whole point.** `append` returns only once the record has
/// reached disk to the extent the policy promises, so the caller updates its in-memory index
/// *after* the call. Reversed, a search could return a record that never reached disk.
///
/// Records are written straight to the file with no `BufWriter` in the way. Buffering in user
/// space would leave bytes that `Never` claims survive a process crash sitting where a process
/// crash destroys them, quietly turning that mode's guarantee into nothing.
pub struct WalWriter {
    file: File,
    path: PathBuf,
    next_seq: u64,
    policy: SyncPolicy,
    unsynced: u32,
    /// Reused so appending does not allocate per record.
    buffer: Vec<u8>,
    len: u64,
}

impl WalWriter {
    /// Starts a fresh log, replacing anything at `path`.
    ///
    /// Used by `anka checkpoint`: once a snapshot is durable, the records it contains are no
    /// longer needed. `first_seq` is where numbering continues, so sequence numbers stay
    /// monotonic across the truncation and a stale log cannot be mistaken for a current one.
    pub fn create(path: &Path, policy: SyncPolicy, first_seq: u64) -> Result<Self, WalError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| WalError::io(path, e))?;

        let mut header = [0u8; HEADER_BYTES];
        header[..4].copy_from_slice(&MAGIC);
        header[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        file.write_all(&header).map_err(|e| WalError::io(path, e))?;
        file.sync_all().map_err(|e| WalError::io(path, e))?;
        // The file may be new, so its directory entry has to be durable too.
        crate::fsync::parent_directory(path).map_err(|e| WalError::io(path, e))?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            next_seq: first_seq,
            policy,
            unsynced: 0,
            buffer: Vec::new(),
            len: HEADER_BYTES as u64,
        })
    }

    /// Reopens an existing log for appending, discarding anything after `valid_bytes`.
    ///
    /// This is the last step of recovery: the replayer reports where the intact records end and
    /// what sequence number comes next, and the torn tail is cut off before a single new byte is
    /// written after it. Appending past a torn record would leave a hole that every later
    /// recovery would stop at, silently losing everything beyond it.
    pub fn reopen(
        path: &Path,
        policy: SyncPolicy,
        next_seq: u64,
        valid_bytes: u64,
    ) -> Result<Self, WalError> {
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| WalError::io(path, e))?;
        file.set_len(valid_bytes)
            .map_err(|e| WalError::io(path, e))?;
        file.sync_all().map_err(|e| WalError::io(path, e))?;

        Ok(Self {
            file,
            path: path.to_path_buf(),
            next_seq,
            policy,
            unsynced: 0,
            buffer: Vec::new(),
            len: valid_bytes,
        })
    }

    /// The sequence number the next [`Self::append`] will assign.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Bytes the log occupies.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len <= HEADER_BYTES as u64
    }

    pub fn policy(&self) -> SyncPolicy {
        self.policy
    }

    /// Appends `record` and returns the sequence number it was given.
    pub fn append(&mut self, record: &Record) -> Result<u64, WalError> {
        let encoded = record.encoded_len();
        if encoded > MAX_RECORD_BYTES {
            return Err(WalError::RecordTooLarge {
                bytes: encoded,
                limit: MAX_RECORD_BYTES,
            });
        }

        let seq = self.next_seq;

        // The whole framed record is assembled in one buffer and written with one call. Writing
        // the framing and the payload separately would let a crash land between them, manufacturing
        // exactly the torn record the reader then has to clean up — avoidable by construction.
        // The framing is reserved first because the checksum is not known until the body exists.
        let mut buf = std::mem::take(&mut self.buffer);
        buf.clear();
        buf.reserve(encoded);
        buf.extend_from_slice(&[0u8; FRAME_BYTES]);
        buf.extend_from_slice(&seq.to_le_bytes());
        buf.push(record.op());
        record.write_payload(&mut buf);

        let len = (buf.len() - FRAME_BYTES) as u32;
        let checksum = crc32(&buf[FRAME_BYTES..]);
        buf[..4].copy_from_slice(&len.to_le_bytes());
        buf[4..8].copy_from_slice(&checksum.to_le_bytes());
        debug_assert_eq!(buf.len(), encoded);

        let written = self
            .file
            .write_all(&buf)
            .map_err(|e| WalError::io(&self.path, e));
        self.buffer = buf;
        written?;

        self.len += encoded as u64;
        self.next_seq += 1;
        self.after_append()?;
        Ok(seq)
    }

    /// Forces everything written so far to disk.
    pub fn sync(&mut self) -> Result<(), WalError> {
        // `sync_data` rather than `sync_all`: the log is only ever appended to, and the one piece
        // of metadata that matters — the file's length — is flushed by `fdatasync` because the
        // data cannot be retrieved without it.
        self.file
            .sync_data()
            .map_err(|e| WalError::io(&self.path, e))?;
        self.unsynced = 0;
        Ok(())
    }

    fn after_append(&mut self) -> Result<(), WalError> {
        match self.policy {
            SyncPolicy::Always => self.sync(),
            SyncPolicy::EveryN(n) => {
                self.unsynced += 1;
                if self.unsynced >= n.get() {
                    self.sync()
                } else {
                    Ok(())
                }
            }
            SyncPolicy::Never => Ok(()),
        }
    }
}

impl std::fmt::Debug for WalWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalWriter")
            .field("path", &self.path)
            .field("next_seq", &self.next_seq)
            .field("len", &self.len)
            .field("policy", &self.policy)
            .finish()
    }
}

/// A record read back off the log.
#[derive(Debug, Clone, PartialEq)]
pub struct Framed {
    pub seq: u64,
    pub record: Record,
    /// Bytes this record occupied, framing included.
    pub bytes: usize,
}

/// Why the log stops here.
///
/// All three mean the same thing to a reader — the intact records end at this offset — and are
/// kept apart so a crash test can assert *which* kind of interruption it produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Torn {
    /// Fewer than the eight framing bytes remain: the write died before `len` was complete.
    ShortFrame { remaining: usize },
    /// `len` describes a record no writer could have produced — too small to hold even `seq` and
    /// `op`, or past the sanity bound. A half-written `len` field looks like this.
    ImpossibleLength { len: usize },
    /// `len` promises more bytes than the file holds: the payload was cut off.
    ShortPayload { needed: usize, remaining: usize },
    /// Everything is present and hashes to something else: a partially-written or damaged record.
    BadChecksum { stored: u32, computed: u32 },
}

/// What reading one record produced.
#[derive(Debug, Clone, PartialEq)]
pub enum Next {
    Record(Framed),
    /// The log ends here, cleanly. Everything from this offset on is discarded.
    Torn(Torn),
}

/// Validates a log header and returns the offset the first record starts at.
pub fn read_header(bytes: &[u8]) -> Result<usize, WalError> {
    if bytes.len() < HEADER_BYTES {
        return Err(WalError::TooShort {
            needed: HEADER_BYTES,
            got: bytes.len(),
        });
    }
    let magic: [u8; 4] = bytes[..4].try_into().expect("4 bytes");
    if magic != MAGIC {
        return Err(WalError::BadMagic { found: magic });
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
    if version != FORMAT_VERSION {
        return Err(WalError::UnsupportedVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    Ok(HEADER_BYTES)
}

/// Reads the record starting at the front of `bytes`.
///
/// `dim` comes from the collection, and the payload arithmetic has to land on it exactly.
pub fn read_record(bytes: &[u8], dim: usize) -> Result<Next, WalError> {
    if bytes.len() < FRAME_BYTES {
        return Ok(Next::Torn(Torn::ShortFrame {
            remaining: bytes.len(),
        }));
    }

    let len = u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes")) as usize;
    let stored = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));

    // A `len` that no writer could have produced is corruption, not a long record. Treating it as
    // a torn tail is right: the usual cause is a `len` field that was only partly written.
    if !(PREFIX_BYTES..=MAX_RECORD_BYTES).contains(&len) {
        return Ok(Next::Torn(Torn::ImpossibleLength { len }));
    }
    if bytes.len() - FRAME_BYTES < len {
        return Ok(Next::Torn(Torn::ShortPayload {
            needed: len,
            remaining: bytes.len() - FRAME_BYTES,
        }));
    }

    let body = &bytes[FRAME_BYTES..FRAME_BYTES + len];
    let computed = crc32(body);
    if computed != stored {
        return Ok(Next::Torn(Torn::BadChecksum { stored, computed }));
    }

    // Past this point the bytes are known intact, so anything that does not parse is a log this
    // build cannot read — an error, never a silent truncation.
    let seq = u64::from_le_bytes(body[..8].try_into().expect("8 bytes"));
    let op = body[8];
    let payload = &body[PREFIX_BYTES..];

    let record = match op {
        OP_INSERT => decode_insert(seq, payload, dim)?,
        OP_DELETE => Record::Delete {
            external_id: payload_u64(seq, op, payload)?,
        },
        OP_CHECKPOINT => Record::Checkpoint {
            snapshot_wal_seq: payload_u64(seq, op, payload)?,
        },
        other => return Err(WalError::UnknownOp { seq, op: other }),
    };

    Ok(Next::Record(Framed {
        seq,
        record,
        bytes: FRAME_BYTES + len,
    }))
}

fn decode_insert(seq: u64, payload: &[u8], dim: usize) -> Result<Record, WalError> {
    if dim == 0 || dim > MAX_DIM {
        return Err(WalError::MalformedPayload {
            seq,
            op: OP_INSERT,
            reason: "the collection's dimension is out of range",
        });
    }
    let fixed = size_of::<u64>() + 1 + dim * size_of::<f32>() + 4;
    if payload.len() < fixed {
        return Err(WalError::MalformedPayload {
            seq,
            op: OP_INSERT,
            reason: "payload is shorter than one vector at the collection's dimension",
        });
    }

    let external_id = u64::from_le_bytes(payload[..8].try_into().expect("8 bytes"));
    let level = payload[8];

    let floats = &payload[9..9 + dim * size_of::<f32>()];
    // Decoded four bytes at a time rather than cast: a record starts at an arbitrary file offset,
    // so the vector inside it is not four-byte aligned and a cast would panic. Replay is bounded
    // by the records since the last checkpoint, not by the collection, so this is not the loop
    // that has to be fast.
    let vector: Vec<f32> = floats
        .chunks_exact(size_of::<f32>())
        .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
        .collect();

    let meta_start = 9 + dim * size_of::<f32>();
    let meta_len = u32::from_le_bytes(
        payload[meta_start..meta_start + 4]
            .try_into()
            .expect("4 bytes"),
    ) as usize;
    let metadata = payload[meta_start + 4..].to_vec();
    if metadata.len() != meta_len {
        return Err(WalError::MalformedPayload {
            seq,
            op: OP_INSERT,
            reason: "metadata length disagrees with the bytes that follow it",
        });
    }

    Ok(Record::Insert {
        external_id,
        level,
        vector,
        metadata,
    })
}

fn payload_u64(seq: u64, op: u8, payload: &[u8]) -> Result<u64, WalError> {
    if payload.len() != size_of::<u64>() {
        return Err(WalError::MalformedPayload {
            seq,
            op,
            reason: "payload is not exactly one 8-byte identifier",
        });
    }
    Ok(u64::from_le_bytes(payload.try_into().expect("8 bytes")))
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    const DIM: usize = 4;

    fn insert(id: u64, level: u8) -> Record {
        Record::Insert {
            external_id: id,
            level,
            vector: (0..DIM).map(|i| id as f32 + i as f32 * 0.25).collect(),
            metadata: Vec::new(),
        }
    }

    /// Writes `records` to a fresh log and hands back the raw file.
    fn log_with(records: &[Record], policy: SyncPolicy) -> (TempDir, PathBuf, Vec<u8>) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wal.log");
        let mut writer = WalWriter::create(&path, policy, 1).unwrap();
        for record in records {
            writer.append(record).unwrap();
        }
        writer.sync().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, writer.len());
        (dir, path, bytes)
    }

    /// Reads every record in `bytes`, stopping at the first torn one.
    fn read_all(bytes: &[u8]) -> (Vec<Framed>, Option<Torn>, usize) {
        let mut offset = read_header(bytes).unwrap();
        let mut records = Vec::new();
        loop {
            match read_record(&bytes[offset..], DIM).unwrap() {
                Next::Record(framed) => {
                    offset += framed.bytes;
                    records.push(framed);
                }
                Next::Torn(torn) => return (records, Some(torn), offset),
            }
        }
    }

    #[test]
    fn every_record_kind_round_trips() {
        let written = vec![
            insert(7, 3),
            Record::Delete { external_id: 7 },
            Record::Checkpoint {
                snapshot_wal_seq: 42,
            },
            Record::Insert {
                external_id: 9,
                level: 0,
                vector: vec![1.0, -2.5, 0.0, f32::MIN_POSITIVE],
                metadata: b"{\"colour\":\"red\"}".to_vec(),
            },
        ];
        let (_dir, _path, bytes) = log_with(&written, SyncPolicy::Always);

        let (read, torn, _) = read_all(&bytes);
        assert_eq!(read.len(), written.len());
        for (index, framed) in read.iter().enumerate() {
            assert_eq!(framed.seq, index as u64 + 1, "sequence numbers are dense");
            assert_eq!(framed.record, written[index]);
        }
        // The log ends where the records do, with nothing left to frame.
        assert_eq!(torn, Some(Torn::ShortFrame { remaining: 0 }));
    }

    #[test]
    fn the_header_is_validated() {
        let (_dir, _path, mut bytes) = log_with(&[insert(1, 0)], SyncPolicy::Always);

        assert!(matches!(
            read_header(&bytes[..HEADER_BYTES - 1]),
            Err(WalError::TooShort { .. })
        ));

        let mut foreign = bytes.clone();
        foreign[..4].copy_from_slice(b"XXXX");
        assert!(matches!(
            read_header(&foreign),
            Err(WalError::BadMagic { .. })
        ));

        bytes[4..8].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        assert!(matches!(
            read_header(&bytes),
            Err(WalError::UnsupportedVersion { .. })
        ));
    }

    /// Scenario 1 of four: the write died before `len` was complete.
    #[test]
    fn a_truncated_length_field_ends_the_log_cleanly() {
        let (_dir, _path, bytes) = log_with(&[insert(1, 0), insert(2, 1)], SyncPolicy::Always);
        let first = read_all(&bytes).0[0].bytes;

        // Zero, one, ... seven bytes of the second record's framing survived.
        for kept in 0..FRAME_BYTES {
            let end = HEADER_BYTES + first + kept;
            let (records, torn, offset) = read_all(&bytes[..end]);
            assert_eq!(records.len(), 1, "kept {kept} framing bytes");
            assert_eq!(torn, Some(Torn::ShortFrame { remaining: kept }));
            assert_eq!(offset, HEADER_BYTES + first);
        }
    }

    /// Still scenario 1, in its other shape: `len` is present but partly written, so it describes
    /// a record that could not exist.
    #[test]
    fn an_impossible_length_ends_the_log_cleanly() {
        let (_dir, _path, mut bytes) = log_with(&[insert(1, 0), insert(2, 1)], SyncPolicy::Always);
        let first = read_all(&bytes).0[0].bytes;
        let len_at = HEADER_BYTES + first;

        for len in [0u32, 1, PREFIX_BYTES as u32 - 1, u32::MAX] {
            bytes[len_at..len_at + 4].copy_from_slice(&len.to_le_bytes());
            let (records, torn, _) = read_all(&bytes);
            assert_eq!(records.len(), 1);
            assert_eq!(
                torn,
                Some(Torn::ImpossibleLength { len: len as usize }),
                "len {len}"
            );
        }
    }

    /// Scenario 2: the framing is complete but the payload was cut off.
    #[test]
    fn a_truncated_payload_ends_the_log_cleanly() {
        let (_dir, _path, bytes) = log_with(&[insert(1, 0), insert(2, 1)], SyncPolicy::Always);
        let (all, _, _) = read_all(&bytes);
        let (first, second) = (all[0].bytes, all[1].bytes);
        let payload = second - FRAME_BYTES;

        for kept in 0..payload {
            let end = HEADER_BYTES + first + FRAME_BYTES + kept;
            let (records, torn, offset) = read_all(&bytes[..end]);
            assert_eq!(records.len(), 1, "kept {kept} payload bytes");
            assert_eq!(
                torn,
                Some(Torn::ShortPayload {
                    needed: payload,
                    remaining: kept
                })
            );
            assert_eq!(offset, HEADER_BYTES + first);
        }
    }

    /// Scenario 3: every byte is present and one of them is wrong.
    #[test]
    fn a_damaged_record_ends_the_log_cleanly() {
        let (_dir, _path, good) = log_with(&[insert(1, 0), insert(2, 1)], SyncPolicy::Always);
        let (all, _, _) = read_all(&good);
        let start = HEADER_BYTES + all[0].bytes;

        // Every byte of the second record, framing included, has to be covered.
        for byte in start..good.len() {
            let mut bytes = good.clone();
            bytes[byte] ^= 0x01;
            let (records, torn, offset) = read_all(&bytes);
            assert!(
                records.len() <= 1,
                "a flipped bit at {byte} was accepted into record {}",
                records.len()
            );
            assert!(torn.is_some(), "a flipped bit at {byte} was not noticed");
            assert_eq!(offset, start, "recovery would resume at the wrong offset");
        }
    }

    /// Damage to a record in the *middle* of the log ends it there, which loses everything after.
    /// That is the correct behaviour and worth pinning: the alternative — scanning forward for a
    /// record that happens to parse — resurrects data around a hole and cannot be reasoned about.
    #[test]
    fn damage_in_the_middle_truncates_everything_after_it() {
        let records: Vec<Record> = (1..=5).map(|i| insert(i, 0)).collect();
        let (_dir, _path, good) = log_with(&records, SyncPolicy::Always);
        let (all, _, _) = read_all(&good);

        let third = HEADER_BYTES + all[0].bytes + all[1].bytes;
        let mut bytes = good.clone();
        bytes[third + FRAME_BYTES + 2] ^= 0xFF;

        let (read, torn, offset) = read_all(&bytes);
        assert_eq!(read.len(), 2);
        assert!(matches!(torn, Some(Torn::BadChecksum { .. })));
        assert_eq!(offset, third);
    }

    /// Intact bytes this build cannot interpret are an error, not a truncation. Treating them as
    /// a torn tail would silently discard records that were successfully committed.
    #[test]
    fn an_unknown_operation_is_an_error_rather_than_a_torn_tail() {
        let (_dir, path, _) = log_with(&[insert(1, 0)], SyncPolicy::Always);
        let mut bytes = std::fs::read(&path).unwrap();

        // Rewrite the op byte and repair the checksum, so the record is intact and unreadable.
        let body = HEADER_BYTES + FRAME_BYTES;
        bytes[body + 8] = 0x7F;
        let len =
            u32::from_le_bytes(bytes[HEADER_BYTES..HEADER_BYTES + 4].try_into().unwrap()) as usize;
        let checksum = crc32(&bytes[body..body + len]);
        bytes[HEADER_BYTES + 4..HEADER_BYTES + 8].copy_from_slice(&checksum.to_le_bytes());

        assert!(matches!(
            read_record(&bytes[HEADER_BYTES..], DIM),
            Err(WalError::UnknownOp { seq: 1, op: 0x7F })
        ));
    }

    /// A log written for one dimension read at another. The arithmetic cannot land, and reporting
    /// that is better than returning a vector made of somebody else's bytes.
    #[test]
    fn a_record_read_at_the_wrong_dimension_is_an_error() {
        let (_dir, _path, bytes) = log_with(&[insert(1, 0)], SyncPolicy::Always);

        assert!(matches!(
            read_record(&bytes[HEADER_BYTES..], DIM + 1),
            Err(WalError::MalformedPayload { seq: 1, .. })
        ));
        assert!(matches!(
            read_record(&bytes[HEADER_BYTES..], 0),
            Err(WalError::MalformedPayload { seq: 1, .. })
        ));
        // Reading narrower leaves bytes over, which the metadata length check catches.
        assert!(matches!(
            read_record(&bytes[HEADER_BYTES..], DIM - 1),
            Err(WalError::MalformedPayload { seq: 1, .. })
        ));
    }

    #[test]
    fn sequence_numbers_continue_across_a_reopen() {
        let (_dir, path, bytes) = log_with(&[insert(1, 0), insert(2, 0)], SyncPolicy::Always);
        let valid = bytes.len() as u64;

        let mut writer = WalWriter::reopen(&path, SyncPolicy::Always, 3, valid).unwrap();
        assert_eq!(writer.next_seq(), 3);
        assert_eq!(writer.append(&insert(3, 0)).unwrap(), 3);

        let (read, _, _) = read_all(&std::fs::read(&path).unwrap());
        assert_eq!(
            read.iter().map(|f| f.seq).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    /// Reopening cuts the torn tail off before writing anything after it. Appending past a torn
    /// record would leave a hole that every later recovery stops at, losing all of it.
    #[test]
    fn reopening_discards_the_torn_tail() {
        let (_dir, path, bytes) = log_with(&[insert(1, 0), insert(2, 0)], SyncPolicy::Always);
        let (all, _, _) = read_all(&bytes);
        let after_first = (HEADER_BYTES + all[0].bytes) as u64;

        // Simulate a crash that left half of the second record behind.
        let mut writer = WalWriter::reopen(&path, SyncPolicy::Always, 2, after_first).unwrap();
        assert_eq!(writer.len(), after_first);
        writer.append(&insert(20, 0)).unwrap();

        let (read, torn, _) = read_all(&std::fs::read(&path).unwrap());
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].seq, 2);
        assert_eq!(
            read[1].record,
            insert(20, 0),
            "the replacement record, not the torn one"
        );
        assert_eq!(torn, Some(Torn::ShortFrame { remaining: 0 }));
    }

    /// The sync policy changes when bytes are forced to disk, never what is written. A log is the
    /// same file under all three, which is what lets the crash tests vary only the policy.
    #[test]
    fn the_sync_policy_does_not_change_the_bytes() {
        let records: Vec<Record> = (1..=4).map(|i| insert(i, i as u8 % 3)).collect();
        let (_a, _pa, always) = log_with(&records, SyncPolicy::Always);
        let (_b, _pb, every) = log_with(&records, SyncPolicy::EveryN(NonZeroU32::new(2).unwrap()));
        let (_c, _pc, never) = log_with(&records, SyncPolicy::Never);

        assert_eq!(always, every);
        assert_eq!(always, never);
    }

    #[test]
    fn an_oversized_record_is_refused() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wal.log");
        let mut writer = WalWriter::create(&path, SyncPolicy::Never, 1).unwrap();

        let huge = Record::Insert {
            external_id: 1,
            level: 0,
            vector: vec![0.0; DIM],
            metadata: vec![0u8; MAX_RECORD_BYTES],
        };
        assert!(matches!(
            writer.append(&huge),
            Err(WalError::RecordTooLarge { .. })
        ));
        // Refused before anything reached the file, so the log is still just its header.
        assert!(writer.is_empty());
        assert_eq!(writer.next_seq(), 1);
    }

    #[test]
    fn an_empty_log_reads_as_empty() {
        let (_dir, _path, bytes) = log_with(&[], SyncPolicy::Always);
        assert_eq!(bytes.len(), HEADER_BYTES);

        let (records, torn, offset) = read_all(&bytes);
        assert!(records.is_empty());
        assert_eq!(torn, Some(Torn::ShortFrame { remaining: 0 }));
        assert_eq!(offset, HEADER_BYTES);
    }
}
