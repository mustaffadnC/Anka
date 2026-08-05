//! What survives a `kill -9`.
//!
//! The property under test is narrow and is the only one a process kill can establish:
//!
//! > **Every record the writer was told had committed is present after recovery.**
//!
//! "Told" is doing the work. Under [`SyncPolicy::Always`], `Collection::insert` returns only after
//! the record has been `fsync`ed, so a caller that saw it return has an acknowledgement. The child
//! process below prints each id as soon as `insert` returns and flushes immediately; the parent
//! kills it at an arbitrary moment, reads what the child managed to claim, and requires recovery
//! to produce at least that much. Anything the child never got to claim may or may not be there —
//! that is what an interrupted write means, and asserting otherwise would be asserting nothing.
//!
//! **What this does not show, and the reason it is said out loud.** `kill -9` kills a *process*.
//! Bytes that reached the kernel through `write()` live on in the page cache and are readable
//! afterwards regardless of whether anything was flushed — so this test passes under
//! [`SyncPolicy::Never`] too, and passing it is not evidence of durability. Power-loss durability
//! needs fault injection at the block layer, and this project does not claim it. See
//! `docs/RESULTS.md`, section 4.
//!
//! The child is this same test binary, re-executed with `ANKA_CRASH_CHILD` set. That keeps the
//! harness self-contained: no extra binary to build, nothing for CI to wire up.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anka_core::{L2Squared, MetricKind};
use anka_index::{DistanceCounter, HnswParams};
use anka_store::collection::{Collection, log_path, snapshot_path};
use anka_store::wal::SyncPolicy;
use anka_store::{Verify, recovery};
use tempfile::TempDir;

const DIM: usize = 8;
const CHILD_ENV: &str = "ANKA_CRASH_CHILD";
const DIR_ENV: &str = "ANKA_CRASH_DIR";
const POLICY_ENV: &str = "ANKA_CRASH_POLICY";
const MODE_ENV: &str = "ANKA_CRASH_MODE";

fn vector(id: u64) -> Vec<f32> {
    (0..DIM).map(|i| id as f32 * 1.5 + i as f32).collect()
}

// -------------------------------------------------------------------------------------------
// child
// -------------------------------------------------------------------------------------------

/// Runs as the child when the environment says so, and never returns.
///
/// Rust's test harness runs every `#[test]` in this binary, so the child re-entry has to happen
/// before any of them — a `#[ctor]`-style hook is not available, so instead every test calls this
/// first. It is cheap when the variable is absent.
fn maybe_run_as_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let dir = PathBuf::from(std::env::var(DIR_ENV).expect("child needs a directory"));
    let policy = match std::env::var(POLICY_ENV).as_deref() {
        Ok("never") => SyncPolicy::Never,
        _ => SyncPolicy::Always,
    };
    let mode = std::env::var(MODE_ENV).unwrap_or_else(|_| "insert".to_string());

    let mut collection = Collection::create(
        &dir,
        DIM,
        MetricKind::L2Squared,
        HnswParams::default(),
        policy,
    )
    .expect("create");

    let mut searcher = collection.searcher();
    let mut counter = DistanceCounter::new();
    let mut stdout = std::io::stdout();

    for id in 1u64.. {
        if mode == "checkpoint" && id.is_multiple_of(25) {
            collection.checkpoint().expect("checkpoint");
        }
        collection
            .insert::<L2Squared>(id, &vector(id), &mut searcher, &mut counter)
            .expect("insert");

        // Only printed once `insert` has returned, which under `Always` is after the fsync. This
        // line is the acknowledgement the parent holds the collection to.
        writeln!(stdout, "{id}").expect("write");
        stdout.flush().expect("flush");
    }
    unreachable!("the loop above is endless; the parent kills this process");
}

// -------------------------------------------------------------------------------------------
// parent
// -------------------------------------------------------------------------------------------

struct Killed {
    /// The highest id the child claimed before it died.
    acknowledged: u64,
}

/// Starts a child writing into `dir`, kills it once it has claimed `until` records, and reports
/// what it claimed.
///
/// Triggered on a count rather than a stopwatch. A fixed delay makes the test's meaning depend on
/// how fast the machine is — on a slow runner the child might not have written anything worth
/// killing, and the assertions below would pass while proving nothing. Waiting for a specific
/// acknowledgement means the kill always lands on a writer that is demonstrably mid-flight,
/// whatever the machine. The deadline is only a backstop against a child that never starts.
fn run_and_kill(dir: &Path, policy: &str, mode: &str, until: u64) -> Killed {
    let mut child = Command::new(std::env::current_exe().expect("test binary path"))
        .env(CHILD_ENV, "1")
        .env(DIR_ENV, dir)
        .env(POLICY_ENV, policy)
        .env(MODE_ENV, mode)
        // The harness would otherwise try to run the tests; the child re-entry happens first, but
        // this keeps its output clean either way.
        .arg("--test-threads=1")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn child");

    let stdout = child.stdout.take().expect("piped stdout");
    let reader = BufReader::new(stdout);

    // The kill goes out as soon as the `until`-th acknowledgement is read, by which point the
    // child has already started on the next record — so it dies with work in flight.
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut acknowledged = 0u64;
    for line in reader.lines() {
        match line {
            Ok(line) => {
                if let Ok(id) = line.trim().parse::<u64>() {
                    acknowledged = acknowledged.max(id);
                }
            }
            Err(_) => break,
        }
        if acknowledged >= until || Instant::now() > deadline {
            break;
        }
    }

    kill_9(&mut child);
    Killed { acknowledged }
}

fn kill_9(child: &mut Child) {
    // SIGKILL, not `Child::kill`'s politeness — the point is a process that gets no chance to
    // flush, close a file, or run a destructor.
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGKILL);
    }
    let _ = child.wait();
}

/// Recovers `dir` and returns how many records came back.
fn recover(dir: &Path) -> (usize, anka_store::Tail) {
    let recovered = recovery::open(&snapshot_path(dir), &log_path(dir), Verify::Body)
        .expect("recovery must not fail on a killed writer");
    recovered
        .index
        .validate()
        .expect("the recovered graph is valid");
    (recovered.index.len(), recovered.tail)
}

/// Printed so a passing run still shows what actually happened. A crash test that says only "ok"
/// hides whether it killed a busy writer or an idle one, and the numbers go into `RESULTS.md`.
fn report(what: &str, acknowledged: u64, recovered: usize, tail: anka_store::Tail) {
    eprintln!("  {what}: {acknowledged} acknowledged, {recovered} recovered, tail {tail:?}");
}

// -------------------------------------------------------------------------------------------
// tests
// -------------------------------------------------------------------------------------------

/// The core claim: under `fsync=always`, nothing acknowledged is lost to a process kill.
#[test]
fn every_acknowledged_record_survives_a_kill() {
    maybe_run_as_child();

    let dir = TempDir::new().unwrap();
    let killed = run_and_kill(dir.path(), "always", "insert", 500);
    assert!(
        killed.acknowledged >= 500,
        "the child only acknowledged {} records — too few to conclude anything",
        killed.acknowledged
    );

    let (recovered, tail) = recover(dir.path());
    report("fsync=always", killed.acknowledged, recovered, tail);
    assert!(
        recovered as u64 >= killed.acknowledged,
        "recovered {recovered} records, but {} had been acknowledged",
        killed.acknowledged
    );
    // At most one more: the record being written when the kill landed may have reached disk
    // without the child getting to claim it.
    assert!(
        recovered as u64 <= killed.acknowledged + 1,
        "recovered {recovered} records against {} acknowledged — more than one unclaimed",
        killed.acknowledged
    );
}

/// A kill during checkpointing, which is where a collection has two files to keep in step and the
/// most ways to be caught halfway.
#[test]
fn a_kill_during_checkpointing_recovers() {
    maybe_run_as_child();

    let dir = TempDir::new().unwrap();
    // Well past several checkpoints, which the child takes every 25 inserts.
    let killed = run_and_kill(dir.path(), "always", "checkpoint", 120);
    assert!(killed.acknowledged >= 120, "no checkpoint was reached");

    let (recovered, tail) = recover(dir.path());
    report("checkpointing", killed.acknowledged, recovered, tail);
    assert!(
        recovered as u64 >= killed.acknowledged,
        "recovered {recovered} records, but {} had been acknowledged across checkpoints",
        killed.acknowledged
    );
}

/// The same test under `fsync=never`, and it passes — which is the point.
///
/// A process kill leaves the page cache intact, so bytes that reached the kernel are still
/// readable. Recovery works, and that says nothing whatsoever about surviving a power cut. Pinned
/// as a test so the distinction stays visible instead of living only in a comment.
#[test]
fn recovery_also_works_without_fsync_which_proves_less_than_it_looks() {
    maybe_run_as_child();

    let dir = TempDir::new().unwrap();
    let killed = run_and_kill(dir.path(), "never", "insert", 500);
    assert!(killed.acknowledged >= 500);

    let (recovered, tail) = recover(dir.path());
    report("fsync=never", killed.acknowledged, recovered, tail);
    assert!(
        recovered as u64 >= killed.acknowledged,
        "recovered {recovered} against {} acknowledged",
        killed.acknowledged
    );
}

/// A process kill cannot tear a record, and this counts how often it fails to.
///
/// SIGKILL does not interrupt a `write()` in progress: the kernel completes the syscall and
/// delivers the signal afterwards. Since a record is assembled in one buffer and issued as a
/// single `write_all`, a process crash lands *between* records and never inside one. That is not
/// an accident of timing but a consequence of how the append is written, and it is worth counting
/// rather than assuming — a future change that split the write into framing-then-payload would
/// start producing torn tails here.
///
/// So the torn-tail handling in `recovery` is not exercised by any process kill, however brutal.
/// It exists for power loss and media failure, and it is tested exhaustively against synthetic
/// damage instead: every truncation point of a record, every single-bit flip.
#[test]
fn a_process_kill_lands_between_records_never_inside_one() {
    maybe_run_as_child();

    let rounds = 8;
    let mut torn = 0;
    for round in 0..rounds {
        let dir = TempDir::new().unwrap();
        let killed = run_and_kill(dir.path(), "always", "insert", 200);
        let (recovered, tail) = recover(dir.path());

        assert!(killed.acknowledged >= 200, "round {round} wrote too little");
        assert!(recovered as u64 >= killed.acknowledged);
        if tail.truncated() {
            torn += 1;
            eprintln!("  round {round}: torn — {tail:?}");
        }
    }
    eprintln!("  {torn} of {rounds} kills produced a torn record");
    assert_eq!(
        torn, 0,
        "a process kill tore a record, which means the append is no longer one write syscall"
    );
}

/// Killing a writer and restarting it repeatedly. Each restart has to pick up a log the previous
/// crash left, repair it, and keep going without losing what came before.
#[test]
fn a_collection_survives_being_killed_repeatedly() {
    maybe_run_as_child();

    let dir = TempDir::new().unwrap();
    let mut previous = 0usize;

    for round in 0..3 {
        // The child recreates the collection, so this exercises repair-then-write on a directory
        // that already holds a torn log from the round before.
        let killed = run_and_kill(dir.path(), "always", "insert", 300);
        let (recovered, tail) = recover(dir.path());
        report(
            &format!("restart {round}"),
            killed.acknowledged,
            recovered,
            tail,
        );

        assert!(
            recovered as u64 >= killed.acknowledged,
            "round {round}: recovered {recovered}, acknowledged {}",
            killed.acknowledged
        );
        assert!(
            recovered > 0,
            "round {round}: recovery produced an empty index"
        );
        previous = previous.max(recovered);
    }

    assert!(previous > 0);
}
