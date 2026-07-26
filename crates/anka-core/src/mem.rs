//! Process memory statistics, for the memory columns in `docs/RESULTS.md`.

/// Resident set size of the current process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Current resident set size (`VmRSS`).
    pub current_bytes: u64,
    /// Peak resident set size since process start (`VmHWM`).
    pub peak_bytes: u64,
}

/// Reads the process's resident set size.
///
/// Linux only, via `/proc/self/status`. Returns `None` everywhere else rather than
/// approximating: the benchmark environment is Linux (see `docs/DESIGN.md`, section 9), and
/// in a document whose entire value rests on its numbers being real, a wrong number is worse
/// than a missing one.
#[cfg(target_os = "linux")]
pub fn resident_set_size() -> Option<MemoryUsage> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    Some(MemoryUsage {
        current_bytes: parse_kb_field(&status, "VmRSS:")?,
        peak_bytes: parse_kb_field(&status, "VmHWM:")?,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn resident_set_size() -> Option<MemoryUsage> {
    None
}

/// Pulls `123456` out of a `/proc/self/status` line such as `VmRSS:\t  123456 kB` and returns
/// it in bytes.
///
/// Compiled and unit-tested on every target even though only Linux calls it, so it cannot rot
/// while development happens on a machine that cannot exercise the caller.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_kb_field(status: &str, field: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with(field))?;
    let kilobytes: u64 = line[field.len()..]
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    kilobytes.checked_mul(1024)
}

/// Formats a byte count for human consumption, e.g. `488.3 MiB`.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
Name:\tanka
State:\tR (running)
VmPeak:\t  1234567 kB
VmSize:\t  1234000 kB
VmHWM:\t    532480 kB
VmRSS:\t    524288 kB
Threads:\t1
";

    #[test]
    fn parses_status_fields_into_bytes() {
        assert_eq!(parse_kb_field(SAMPLE, "VmRSS:"), Some(524_288 * 1024));
        assert_eq!(parse_kb_field(SAMPLE, "VmHWM:"), Some(532_480 * 1024));
    }

    /// `VmSize` is a prefix of nothing, but `VmHWM` and `VmPeak` sit next to each other in the
    /// file — a sloppy `contains` match would confuse them.
    #[test]
    fn matches_the_exact_field() {
        assert_eq!(parse_kb_field(SAMPLE, "VmPeak:"), Some(1_234_567 * 1024));
        assert_eq!(parse_kb_field(SAMPLE, "VmFoo:"), None);
    }

    #[test]
    fn malformed_field_yields_none() {
        assert_eq!(parse_kb_field("VmRSS:\tnot-a-number kB", "VmRSS:"), None);
        assert_eq!(parse_kb_field("VmRSS:", "VmRSS:"), None);
    }

    #[test]
    fn formats_byte_counts() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(512 * 1024 * 1024), "512.0 MiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    /// On Linux this must produce real numbers; elsewhere it must admit it cannot.
    #[test]
    fn resident_set_size_matches_the_platform() {
        let usage = resident_set_size();
        if cfg!(target_os = "linux") {
            let usage = usage.expect("/proc/self/status should be readable on Linux");
            assert!(usage.current_bytes > 0);
            assert!(usage.peak_bytes >= usage.current_bytes);
        } else {
            assert!(usage.is_none());
        }
    }
}
