//! Making a directory entry durable.
//!
//! Shared by the snapshot writer and the write-ahead log, because both create files whose
//! *existence* has to survive a crash, not just whose contents do.

use std::fs::File;
use std::path::Path;

/// Flushes the directory holding `path`.
///
/// Creating or renaming a file writes an entry in its directory, and that entry lives in the
/// directory's own metadata. `fsync` on the file makes its *contents* durable and says nothing
/// about the entry pointing at it, so a crash can leave a fully-written file that no longer has a
/// name. This is the step that is most often left out, and the one whose absence is invisible
/// until a machine loses power.
#[cfg(unix)]
pub(crate) fn parent_directory(path: &Path) -> std::io::Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    let dir = parent.unwrap_or_else(|| Path::new("."));
    File::open(dir)?.sync_all()
}

/// Windows has no directory handle to sync, and `rename` over an existing file is not atomic there
/// either. The project builds and measures on Linux; on other targets files are written correctly
/// but their durability across a crash is **not** claimed. See `docs/RESULTS.md`, section 4.
#[cfg(not(unix))]
pub(crate) fn parent_directory(_path: &Path) -> std::io::Result<()> {
    Ok(())
}
