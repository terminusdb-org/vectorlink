//! Vector index configuration (IVF_PQ parameters).

/// Configuration for the vector ANN index (IVF_PQ).
/// Parameters are pinned as constants — changing them perturbs ranking
/// and should be treated like a model version bump.
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Number of IVF partitions. More partitions = faster search at the cost of
    /// index build time and recall. Recommended: sqrt(n) for corpus size n.
    pub num_partitions: usize,
    /// Number of PQ sub-vectors. Must divide the embedding dimension evenly.
    pub num_sub_vectors: usize,
    /// Number of probes during search (how many partitions to scan).
    /// Higher = better recall, slower search.
    pub nprobes: usize,
    /// Refine factor: re-rank this many candidates with full-precision vectors.
    /// Higher = better recall at the cost of latency. None = no refinement.
    pub refine_factor: Option<u32>,
}

impl VectorIndexConfig {
    /// Sane defaults for a given embedding dimension.
    /// These are pinned — treat changes as a model bump.
    ///
    /// INVARIANT: `num_sub_vectors` always divides `dim` evenly (PQ requirement).
    /// If `dim` is not evenly divisible by the target sub-vector count, we find
    /// the largest divisor of `dim` that is <= the target. This guarantees the
    /// index build never fails due to dimension/sub-vector mismatch.
    ///
    /// # Panics
    ///
    /// Panics if `dim == 0`. This is called at service startup; a zero-dim
    /// embedding is a configuration error that must fail loud at boot.
    pub fn default_for_dim(dim: usize) -> Self {
        assert!(dim > 0, "embedding dimension must be > 0 (check TDB_SEARCH_DIM)");

        // Target sub-vector counts by dimension range:
        // 128-d → target 16 (8 dims per sub-vector)
        // 256-d → target 32 (8 dims each)
        // 768-d → target 48 (16 dims each)
        // >768  → target dim/16
        let target = match dim {
            d if d <= 128 => (d / 8).max(1),
            d if d <= 256 => (d / 8).max(1),
            d if d <= 768 => (d / 16).max(1),
            d => (d / 16).max(1),
        };

        // Find the largest divisor of `dim` that is <= target.
        // This guarantees dim % num_sub_vectors == 0 (PQ requirement).
        let num_sub_vectors = largest_divisor_leq(dim, target);

        Self {
            // Start with 16 partitions; scale up with corpus via
            // `recommended_num_partitions` when corpus size is known.
            num_partitions: 16,
            num_sub_vectors,
            nprobes: 8,
            refine_factor: Some(10),
        }
    }
}

/// Find the largest divisor of `n` that is <= `target`.
/// Returns 1 if no divisor in [2, target] exists (1 always divides anything).
///
/// Used to ensure `dim % num_sub_vectors == 0` for PQ index creation.
pub(super) fn largest_divisor_leq(n: usize, target: usize) -> usize {
    // Search downward from target to find a divisor of n.
    // For typical embedding dimensions (128, 256, 384, 512, 768, 1024, 1536)
    // this terminates quickly because they have many small factors.
    let mut candidate = target;
    while candidate > 1 {
        if n.is_multiple_of(candidate) {
            return candidate;
        }
        candidate -= 1;
    }
    // 1 always divides any positive number.
    1
}
