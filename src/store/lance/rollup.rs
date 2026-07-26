// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

#![forbid(unsafe_code)]

//! Incremental index delta cascade, analogous to TerminusDB's
//! `exponential_rollup_strategy/1` (in `src/core/api/api_optimize.pl`).
//!
//! TerminusDB rolls up layer stacks in power-of-base partitions (3, 9, 27, 81…)
//! to reduce O(N) layer traversal to O(log₃(N)). We apply the same strategy to
//! LanceDB index deltas: each push creates a small delta index via
//! `optimize_indices(append())`, and over many pushes these accumulate. Search
//! at HEAD probes all deltas sequentially, causing linear latency growth.
//!
//! The incremental cascade merges deltas in base-3 groups after each push,
//! maintaining a hierarchy of consolidated indices at HEAD. Historical tagged
//! versions retain their original fragmented indices (temporal search contract),
//! except during compaction when intermediate tags are retagged to boundary
//! versions (transactional retagging within the compaction lock).

use lance::dataset::Dataset;
use lance::index::DatasetIndexExt;
use lance_index::optimize::OptimizeOptions;

use crate::kernel::error::StoreError;

/// Determine whether a commit at the given 0-indexed position should trigger
/// index creation. Indices are created at every 3rd commit: positions 2, 5, 8,
/// 11, 14, ... (0-indexed). Non-indexed commits rely on KNN fallback for search.
pub fn should_index_commit(commit_position: usize) -> bool {
    (commit_position + 1).is_multiple_of(3)
}

/// Integer logarithm: floor(log_base(N)).
/// Returns 0 when base > N.
fn ilog(n: usize, base: usize) -> usize {
    if base > n {
        return 0;
    }
    let mut exp = 0;
    let mut val = n;
    while val >= base {
        val /= base;
        exp += 1;
    }
    exp
}

/// Compute exponential roll-up partitions (base 3 by default).
///
/// Direct port of TerminusDB's `positions/4` predicate from
/// `src/core/api/api_optimize.pl`:
///
/// ```prolog
/// positions(Length, Base, Start, End) :-
///     Length >= Base,
///     log(Length, Base, Exp),
///     Offset is Base ** Exp,
///     (   Start = 0, End is Offset - 1
///     ;   Offset < Length,
///         New_Length is Length - Offset,
///         positions(New_Length, Base, Shifted_Start, Shifted_End),
///         Start is Shifted_Start + Offset,
///         End is Shifted_End + Offset).
/// ```
///
/// Returns a list of (start, end) inclusive index pairs for partitions of
/// size >= base. Partitions are ordered from largest to smallest (oldest
/// data first), matching TerminusDB's stack order [Oldest, ..., Newest].
///
/// Example with base=3, total=13:
/// - ilog(13, 3) = 2, offset = 3^2 = 9 → partition [0, 8] (size 9)
/// - remaining = 13 - 9 = 4
/// - ilog(4, 3) = 1, offset = 3^1 = 3 → partition [9, 11] (size 3)
/// - remaining = 4 - 3 = 1
/// - 1 < 3, no more partitions
/// - Result: [(0, 8), (9, 11)]
pub fn rollup_partitions(total: usize, base: usize) -> Vec<(usize, usize)> {
    if total < base {
        return Vec::new();
    }

    let mut partitions = Vec::new();
    let mut offset = 0usize;
    let mut remaining = total;

    while remaining >= base {
        let exp = ilog(remaining, base);
        let size = base.pow(exp as u32);
        partitions.push((offset, offset + size - 1));
        offset += size;
        remaining -= size;
    }

    partitions
}

/// Exponential index roll-up: merge index deltas in power-of-base groups.
///
/// Given N index deltas at HEAD, computes partitions using base 3:
/// e.g. N=123 → [(0, 80), (81, 107), (108, 116), (117, 119), (120, 122)]
///
/// Then calls `optimize_indices(merge(partition_size))` for each partition,
/// from smallest to largest. Merging smallest first means each subsequent
/// larger merge includes the previously merged results, building the
/// exponential hierarchy naturally.
///
/// Each merge creates a new dataset version with a consolidated index.
/// Historical tagged versions retain their original fragmented indices
/// (temporal search contract).
///
/// Returns (indices_before, indices_after) for observability.
pub async fn io_exponential_rollup(
    ds: &mut Dataset,
    base: usize,
) -> Result<(usize, usize), StoreError> {
    let indices_before = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;

    if indices_before.is_empty() {
        return Ok((0, 0));
    }

    let count_before = indices_before.len();

    // Compute partitions. LanceDB's merge(N) merges the N most recent delta
    // indices. We merge from the largest partition (oldest indices) to the
    // smallest (newest), so each merge consolidates a progressively larger
    // chunk of the history.
    let partitions = rollup_partitions(count_before, base);

    if partitions.is_empty() {
        return Ok((count_before, count_before));
    }

    eprintln!(
        "[rollup] {} indices, base={}, partitions={:?}",
        count_before, base, partitions
    );

    // Merge from largest partition to smallest. LanceDB's merge(N) takes the
    // N most recent indices, so we merge the largest group first (which covers
    // the oldest indices), then progressively smaller groups.
    for (start, end) in partitions.iter().rev() {
        let size = end - start + 1;
        eprintln!("[rollup] merging {} indices (range {}..{})", size, start, end);
        ds.optimize_indices(&OptimizeOptions::merge(size))
            .await
            .map_err(|e| {
                StoreError::Internal(format!("rollup merge({}) failed: {}", size, e))
            })?;
    }

    let indices_after = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices after rollup failed: {}", e)))?;

    let count_after = indices_after.len();

    eprintln!(
        "[rollup] indices {}→{} ({} merges)",
        count_before,
        count_after,
        partitions.len()
    );

    Ok((count_before, count_after))
}

/// Compute how many merge(3) calls are needed for a given delta count.
///
/// Each merge(3) promotes 3 indices at level N to 1 index at level N+1.
/// The cascade fires when delta_count is divisible by 3, 9, 27, etc.
/// - delta_count=3 → 1 merge (3 deltas → 1 level-1 index)
/// - delta_count=9 → 2 merges (3 level-1 → 1 level-2)
/// - delta_count=27 → 3 merges (3 level-2 → 1 level-3)
/// - delta_count=1,2,4,5,7,8 → 0 merges (not divisible by 3)
pub fn merges_needed(delta_count: u64) -> usize {
    let mut count = 0;
    let mut n = delta_count;
    while n > 0 && n.is_multiple_of(3) {
        count += 1;
        n /= 3;
    }
    count
}

/// Incremental merge(3) cascade: maintain a base-3 index hierarchy at HEAD.
///
/// After each push, `optimize_indices(append())` creates a small delta index.
/// This function calls `merge(3)` `merges_needed(delta_count)` times. Each
/// `merge(3)` merges the 3 most recent indices, and the cascade builds
/// naturally: after the first merge, the 3 most recent are level-N indices,
/// so the next merge promotes them to level-(N+1).
///
/// Each `merge(3)` call is guarded by `load_indices().len() >= 3` to handle
/// edge cases where delta_count doesn't perfectly match the actual index count
/// (e.g., no-op pushes that don't create deltas).
///
/// Returns (indices_before, indices_after) for observability.
pub async fn io_incremental_cascade(
    ds: &mut Dataset,
    delta_count: u64,
) -> Result<(usize, usize), StoreError> {
    let indices_before = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices failed: {}", e)))?;

    let count_before = indices_before.len();
    let needed = merges_needed(delta_count);

    if needed == 0 {
        return Ok((count_before, count_before));
    }

    eprintln!(
        "[cascade] {} indices, delta_count={}, merges_needed={}",
        count_before, delta_count, needed
    );

    for i in 0..needed {
        let indices = ds
            .load_indices()
            .await
            .map_err(|e| StoreError::Internal(format!("load indices in cascade failed: {}", e)))?;

        if indices.len() < 3 {
            eprintln!(
                "[cascade] stopping at merge {}/{} — only {} indices < 3",
                i + 1, needed, indices.len()
            );
            break;
        }

        eprintln!(
            "[cascade] merge {}/{} ({} indices → merge(3))",
            i + 1, needed, indices.len()
        );

        ds.optimize_indices(&OptimizeOptions::merge(3))
            .await
            .map_err(|e| StoreError::Internal(format!("cascade merge(3) failed: {}", e)))?;
    }

    let indices_after = ds
        .load_indices()
        .await
        .map_err(|e| StoreError::Internal(format!("load indices after cascade failed: {}", e)))?;

    let count_after = indices_after.len();

    eprintln!(
        "[cascade] indices {}→{} ({} merges attempted)",
        count_before, count_after, needed
    );

    Ok((count_before, count_after))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::lance::ChunkRow;
    use crate::store::vector_index::io_ensure_vector_index;
    use crate::store::vector_index::tests::{make_test_config, make_test_store, make_chunk_row};
    use crate::store::lance::io_ensure_fts_index_on_dataset;
    use std::time::Instant;

    /// H1 (go/no-go gate): Verify that `optimize_indices(merge(3))` completes
    /// without deadlock within the same tokio runtime. Also verifies that both
    /// vector and FTS indices are merged together (A5) and measures latency (A4).
    ///
    /// If this test hangs or fails, the entire cascade approach is blocked.
    #[tokio::test]
    async fn merge3_same_runtime() {
        let (store, _tmp) = make_test_store(128);
        let config = make_test_config(128);

        // Insert enough rows for IVF training (needs >= num_partitions rows).
        let rows: Vec<ChunkRow> = (0..300)
            .map(|i| make_chunk_row(&format!("doc/{}", i), 128, i as u64 + 1))
            .collect();
        for row in &rows {
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(row))
                .await
                .expect("upsert");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let mut ds = ds_arc.write().await;

        // Create vector + FTS indices.
        io_ensure_vector_index(&mut ds, &config, false).await.expect("create vector index");
        io_ensure_fts_index_on_dataset(&mut ds).await.expect("create FTS index");

        let indices_after_create = ds.load_indices().await.expect("load indices after create");
        let count_after_create = indices_after_create.len();
        assert!(count_after_create >= 2, "should have vector + FTS indices");

        // Append more data (creates new unindexed fragments).
        drop(ds);
        for i in 300..310 {
            let row = make_chunk_row(&format!("doc/{}", i), 128, i as u64 + 1);
            store
                .io_upsert_chunks("admin/test", "main", &row.doc_id, std::slice::from_ref(&row))
                .await
                .expect("upsert append");
        }

        let ds_arc = store.io_open_dataset("admin/test", "main").await.unwrap();
        let mut ds = ds_arc.write().await;

        // Append: creates delta indices for new fragments.
        io_ensure_vector_index(&mut ds, &config, false).await.expect("append vector index");
        io_ensure_fts_index_on_dataset(&mut ds).await.expect("append FTS index");

        let indices_after_append = ds.load_indices().await.expect("load indices after append");
        let count_after_append = indices_after_append.len();
        eprintln!(
            "[merge3_test] indices after create={}, after append={}",
            count_after_create, count_after_append
        );

        // The critical test: call merge(3) within the same tokio runtime.
        // If this deadlocks, the test will hang and timeout.
        let start = Instant::now();
        ds.optimize_indices(&OptimizeOptions::merge(3))
            .await
            .expect("merge(3) should complete without deadlock");
        let elapsed = start.elapsed();

        let indices_after_merge = ds.load_indices().await.expect("load indices after merge");
        let count_after_merge = indices_after_merge.len();

        eprintln!(
            "[merge3_test] merge(3) completed in {:?}, indices {}→{}→{}",
            elapsed, count_after_create, count_after_append, count_after_merge
        );

        // Verify merge reduced index count (3 merged into 1 for each index type).
        assert!(
            count_after_merge < count_after_append,
            "merge(3) should reduce index count: before={}, after={}",
            count_after_append,
            count_after_merge
        );

        // Verify both vector and FTS indices still exist after merge.
        let has_vector = indices_after_merge.iter().any(|i| i.name == "embedding_ann");
        let has_fts = indices_after_merge.iter().any(|i| i.name == "content_fts");
        assert!(has_vector, "vector index should exist after merge(3)");
        assert!(has_fts, "FTS index should exist after merge(3)");
    }

    #[test]
    fn ilog_basic() {
        assert_eq!(ilog(1, 3), 0);
        assert_eq!(ilog(2, 3), 0);
        assert_eq!(ilog(3, 3), 1);
        assert_eq!(ilog(8, 3), 1);
        assert_eq!(ilog(9, 3), 2);
        assert_eq!(ilog(26, 3), 2);
        assert_eq!(ilog(27, 3), 3);
        assert_eq!(ilog(80, 3), 3);
        assert_eq!(ilog(81, 3), 4);
        assert_eq!(ilog(122, 3), 4);
        assert_eq!(ilog(123, 3), 4);
    }

    #[test]
    fn partitions_empty_when_below_base() {
        assert!(rollup_partitions(0, 3).is_empty());
        assert!(rollup_partitions(1, 3).is_empty());
        assert!(rollup_partitions(2, 3).is_empty());
    }

    #[test]
    fn partitions_exactly_base() {
        let p = rollup_partitions(3, 3);
        assert_eq!(p, vec![(0, 2)]);
    }

    #[test]
    fn partitions_13_elements() {
        // 13 = 9 + 3 + 1 → partitions [0..8], [9..11]
        let p = rollup_partitions(13, 3);
        assert_eq!(p, vec![(0, 8), (9, 11)]);
    }

    #[test]
    fn partitions_27_elements() {
        // 27 = 27 → partition [0..26]
        let p = rollup_partitions(27, 3);
        assert_eq!(p, vec![(0, 26)]);
    }

    #[test]
    fn partitions_40_elements() {
        // 40 = 27 + 9 + 3 + 1 → partitions [0..26], [27..35], [36..38]
        let p = rollup_partitions(40, 3);
        assert_eq!(p, vec![(0, 26), (27, 35), (36, 38)]);
    }

    #[test]
    fn partitions_81_elements() {
        // 81 = 81 → partition [0..80]
        let p = rollup_partitions(81, 3);
        assert_eq!(p, vec![(0, 80)]);
    }

    #[test]
    fn partitions_121_elements() {
        // 121 = 81 + 27 + 9 + 3 + 1 → partitions [0..80], [81..107], [108..116], [117..119]
        let p = rollup_partitions(121, 3);
        assert_eq!(
            p,
            vec![(0, 80), (81, 107), (108, 116), (117, 119)]
        );
    }

    #[test]
    fn partitions_123_elements() {
        // 123 = 81 + 27 + 9 + 3 + 3 → partitions [0..80], [81..107], [108..116], [117..119], [120..122]
        let p = rollup_partitions(123, 3);
        assert_eq!(
            p,
            vec![(0, 80), (81, 107), (108, 116), (117, 119), (120, 122)]
        );
    }

    #[test]
    fn partitions_cover_all_except_remainder() {
        // The union of all partitions should cover total - (total % base^0)
        // i.e., everything except the final remainder that's < base.
        for total in [3, 5, 9, 13, 27, 40, 81, 100, 121, 123, 200] {
            let p = rollup_partitions(total, 3);
            let covered: usize = p.iter().map(|(s, e)| e - s + 1).sum();
            let remainder = total - covered;
            assert!(
                remainder < 3,
                "total={} covered={} remainder={} should be < base",
                total,
                covered,
                remainder
            );
        }
    }

    #[test]
    fn partitions_are_non_overlapping_and_ordered() {
        for total in [3, 9, 13, 27, 40, 81, 100, 121, 123, 200] {
            let p = rollup_partitions(total, 3);
            for i in 1..p.len() {
                assert!(
                    p[i].0 > p[i - 1].1,
                    "total={} partitions overlap or not ordered: {:?}",
                    total,
                    p
                );
            }
        }
    }

    #[test]
    fn merges_needed_divisible_by_3() {
        assert_eq!(merges_needed(3), 1);
        assert_eq!(merges_needed(6), 1);
        assert_eq!(merges_needed(12), 1);
        assert_eq!(merges_needed(15), 1);
    }

    #[test]
    fn merges_needed_divisible_by_9() {
        assert_eq!(merges_needed(9), 2);
        assert_eq!(merges_needed(18), 2);
        assert_eq!(merges_needed(36), 2);
        assert_eq!(merges_needed(45), 2);
    }

    #[test]
    fn merges_needed_divisible_by_27() {
        assert_eq!(merges_needed(27), 3);
        assert_eq!(merges_needed(54), 3);
        assert_eq!(merges_needed(108), 3);
    }

    #[test]
    fn merges_needed_divisible_by_81() {
        assert_eq!(merges_needed(81), 4);
        assert_eq!(merges_needed(243), 5);
    }

    #[test]
    fn merges_needed_not_divisible_by_3() {
        assert_eq!(merges_needed(0), 0);
        assert_eq!(merges_needed(1), 0);
        assert_eq!(merges_needed(2), 0);
        assert_eq!(merges_needed(4), 0);
        assert_eq!(merges_needed(5), 0);
        assert_eq!(merges_needed(7), 0);
        assert_eq!(merges_needed(8), 0);
        assert_eq!(merges_needed(10), 0);
        assert_eq!(merges_needed(13), 0);
        assert_eq!(merges_needed(134), 0);
    }

    #[test]
    fn should_index_commit_at_every_3rd() {
        assert!(should_index_commit(2));
        assert!(should_index_commit(5));
        assert!(should_index_commit(8));
        assert!(should_index_commit(11));
        assert!(should_index_commit(14));
        assert!(should_index_commit(17));
        assert!(should_index_commit(20));
    }

    #[test]
    fn should_index_commit_false_for_non_3rd() {
        assert!(!should_index_commit(0));
        assert!(!should_index_commit(1));
        assert!(!should_index_commit(3));
        assert!(!should_index_commit(4));
        assert!(!should_index_commit(6));
        assert!(!should_index_commit(7));
        assert!(!should_index_commit(9));
        assert!(!should_index_commit(10));
    }
}
