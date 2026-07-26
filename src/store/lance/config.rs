// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 DFRNT AB

//! Vector index configuration (IVF_HNSW_SQ parameters).
//!
//! Index type: IVF_HNSW_SQ (Inverted File + Hierarchical Navigable Small World
//! + Scalar Quantisation). This provides:
//! - IVF layer for scalable partition-based coarse search
//! - HNSW graph for high-recall traversal within partitions
//! - Scalar Quantisation (SQ) for per-dimension precision preservation
//!   (far less lossy than PQ's sub-vector codebook compression)
//!
//! Previous index type was IVF_PQ (8-bit Product Quantisation), which compresses
//! sub-vector groups into codebook centroids — a precision drag for entity
//! resolution where distance ranking fidelity matters.

/// Configuration for the vector ANN index (IVF_HNSW_SQ).
///
/// `num_partitions` is a **minimum floor** — at index creation time, the
/// actual partition count is computed as `max(num_partitions, sqrt(row_count) * 3/4)`
/// so the index scales with corpus size. Search-time parameters (`nprobes`,
/// `refine_factor`) are fixed here because the index already knows its own
/// partition count — these control how much of the index to scan per query.
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Minimum number of IVF partitions. The actual count at creation time is
    /// `max(self, sqrt(row_count) * 3/4)`. 16 is a reasonable floor for small corpora.
    pub num_partitions: usize,
    /// HNSW: maximum number of bi-directional links per node.
    /// Higher M = better recall + more memory. Literature default: 16-64.
    /// We use 30 for high-recall entity resolution.
    pub m: usize,
    /// HNSW: size of the dynamic candidate list during index construction.
    /// Higher = better graph quality at the cost of build time.
    /// Must be >= 2 * M. Literature default: 100-200.
    pub ef_construction: usize,
    /// Number of probes during search (how many IVF partitions to scan).
    /// Higher = better recall, slower search.
    pub nprobes: usize,
    /// Refine factor: re-rank this many candidates with full-precision vectors.
    /// Higher = better recall at the cost of latency. None = no refinement.
    /// With SQ (less lossy than PQ), lower refine factors are acceptable.
    pub refine_factor: Option<u32>,
}

impl VectorIndexConfig {
    /// Sane defaults for a given embedding dimension.
    /// These are pinned — treat changes as a model bump.
    ///
    /// IVF_HNSW_SQ does NOT require `num_sub_vectors` (that was PQ-specific).
    /// SQ quantises each dimension independently (8-bit scalar), so no
    /// divisibility constraint exists.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`. This is called at service startup; a zero-dim
    /// embedding is a configuration error that must fail loud at boot.
    pub fn default_for_dim(dim: usize) -> Self {
        assert!(dim > 0, "embedding dimension must be > 0 (check VECTORLINK_DIM)");
        // dim is used to validate the caller's intent; SQ doesn't need
        // per-dimension configuration beyond ensuring non-zero.
        let _ = dim;

        Self {
            // Minimum floor for IVF partitions. At creation time, the actual
            // count is max(16, sqrt(row_count)) — e.g. sqrt(2M) ≈ 1414.
            // 16 is sufficient for small corpora (<256 rows); larger corpora
            // get sqrt(n) partitions computed at index creation/rebuild time.
            num_partitions: 16,
            // HNSW graph connectivity — 30 edges for high recall.
            // Literature: M=16 is standard; M=30-48 for high-precision workloads.
            m: 30,
            // Construction quality — 200 candidates during build.
            // Higher than default (150) for better graph connectivity.
            ef_construction: 200,
            // IVF probe count for search. With 1024 partitions, probing 4
            // scans ~8k vectors — sufficient for top-5 recall on 2M vectors.
            // Each probe opens an HNSW graph + scans it; keeping this low is
            // the single biggest latency lever for sub-200ms search.
            nprobes: 4,
            // SQ is far less lossy than PQ, so a refine factor of 2 is sufficient
            // for top-k <= 15 (our over-fetch is count*3=15 for count=5).
            // Each refine reads a full-precision vector from disk — keeping it low
            // is critical for sub-200ms latency on 2M vectors.
            refine_factor: Some(2),
        }
    }

    /// Compute the actual number of IVF partitions for a given corpus size.
    ///
    /// Returns `max(self.num_partitions, sqrt(row_count) * 3 / 4)`.
    ///
    /// The classic IVF recommendation is sqrt(n). We use 3/4 * sqrt(n) as a
    /// slight reduction — with IVF_HNSW, the HNSW graph handles intra-partition
    /// search in O(log(n)), so slightly fewer partitions than sqrt(n) reduces
    /// IVF centroid lookup overhead and index file count without significantly
    /// increasing per-partition HNSW traversal cost.
    ///
    /// Empirically validated on 2M vectors: 1060 partitions (3/4 * sqrt(2M))
    /// matches the performance of a fixed 1024, giving ~67ms warm vector search
    /// and ~136ms warm hybrid search.
    ///
    /// Examples:
    /// - 10k rows  → max(16, 75)   = 75 partitions
    /// - 100k rows → max(16, 237)  = 237 partitions
    /// - 2M rows   → max(16, 1060) = 1060 partitions
    /// - 10M rows  → max(16, 2371) = 2371 partitions
    pub fn recommended_num_partitions(&self, row_count: usize) -> usize {
        let sqrt_n = (row_count as f64).sqrt();
        let adaptive = (sqrt_n * 0.75) as usize;
        self.num_partitions.max(adaptive)
    }
}
