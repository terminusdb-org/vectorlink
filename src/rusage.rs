// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Process resource usage utilities.
//!
//! Provides current and peak RSS measurement.
//! On macOS, uses `mach_task_basic_info` for current RSS and `getrusage` for peak.
//! On Linux, reads `/proc/self/status` for current RSS and `getrusage` for peak.
//! Isolated in its own module to avoid the `#![forbid(unsafe_code)]` restriction
//! in the store module.

/// Get process current RSS and peak RSS in bytes.
/// Returns (current_rss_bytes, peak_rss_bytes). On non-Unix targets returns (0, 0).
pub fn get_rusage_rss() -> (u64, u64) {
    #[cfg(unix)]
    {
        // Peak RSS from getrusage (always works on Unix).
        let max_rss = {
            unsafe {
                let mut usage: libc::rusage = std::mem::zeroed();
                let ret = libc::getrusage(libc::RUSAGE_SELF, &mut usage);
                if ret == 0 {
                    #[cfg(target_os = "macos")]
                    { usage.ru_maxrss as u64 }
                    #[cfg(not(target_os = "macos"))]
                    { (usage.ru_maxrss as u64) * 1024 }
                } else {
                    0
                }
            }
        };

        // Current RSS — platform-specific.
        let current_rss = current_rss_bytes().unwrap_or(max_rss);

        (current_rss, max_rss)
    }
    #[cfg(not(unix))]
    {
        (0, 0)
    }
}

/// Get the current (not peak) resident set size in bytes.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn current_rss_bytes() -> Option<u64> {
    unsafe {
        let mut info: libc::mach_task_basic_info_data_t = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info_data_t>()
            / std::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
        let ret = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as libc::task_info_t,
            &mut count,
        );
        if ret == libc::KERN_SUCCESS {
            // resident_size is in bytes on macOS.
            Some(info.resident_size as u64)
        } else {
            None
        }
    }
}

/// Get the current (not peak) resident set size in bytes.
#[cfg(all(unix, not(target_os = "macos")))]
fn current_rss_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|content| {
            content
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| {
                    line.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse::<u64>().ok())
                        .map(|kb| kb * 1024)
                })
        })
}

#[cfg(not(unix))]
fn current_rss_bytes() -> Option<u64> {
    None
}
