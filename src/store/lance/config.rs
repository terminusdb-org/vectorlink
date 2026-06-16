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
/// Parameters are pinned as constants — changing them perturbs ranking
/// and should be treated like a model version bump.
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Number of IVF partitions. More partitions = faster search at the cost of
    /// index build time and recall. Recommended: sqrt(n) for corpus size n.
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
        assert!(dim > 0, "embedding dimension must be > 0 (check TDB_SEARCH_DIM)");
        // dim is used to validate the caller's intent; SQ doesn't need
        // per-dimension configuration beyond ensuring non-zero.
        let _ = dim;

        Self {
            // Start with 16 partitions; scale up with corpus via
            // `recommended_num_partitions` when corpus size is known.
            num_partitions: 16,
            // HNSW graph connectivity — 30 edges for high recall.
            // Literature: M=16 is standard; M=30-48 for high-precision workloads.
            m: 30,
            // Construction quality — 200 candidates during build.
            // Higher than default (150) for better graph connectivity.
            ef_construction: 200,
            // IVF probe count for search.
            nprobes: 8,
            // SQ is far less lossy than PQ, so a refine factor of 5 is sufficient
            // (vs 10 needed with PQ). Still re-ranks with full-precision vectors.
            refine_factor: Some(5),
        }
    }
}
