#![forbid(unsafe_code)]

//! Distance transforms — the single place distance scaling lives.
//! Pure, unit-tested with fixed fixtures.

/// Normalise a Lance cosine distance to the [0, 1] reference scale.
/// Lance cosine distance is in [0, 2] (0 = identical, 2 = opposite).
/// Reference scale: 0 = identical, 0.5 = unrelated/orthogonal, 1 = opposite.
pub fn normalized_cosine_from_lance(lance_cosine_distance: f32) -> f32 {
    (lance_cosine_distance / 2.0).clamp(0.0, 1.0)
}

/// L2-normalise a vector in place (unit length). Used before insertion to ensure
/// cosine distance = inner-product distance after normalisation.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// Compute normalized cosine distance between two L2-normalized vectors.
/// Returns a value on the [0, 1] reference scale (same as `normalized_cosine_from_lance`):
/// 0 = identical, 0.5 = unrelated/orthogonal, 1 = opposite.
///
/// For two L2-normalized vectors a and b:
///   cosine_similarity = dot(a, b)
///   lance_cosine_distance = 1 - cosine_similarity   (in [0, 2])
///   normalized = lance_cosine_distance / 2           (in [0, 1])
///
/// This is the SAME scale that `/search`, `/similar`, `/duplicates`, and `/resolve`
/// return — no new scale invented.
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let lance_distance = 1.0 - dot;
    normalized_cosine_from_lance(lance_distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_distance_is_zero() {
        assert_eq!(normalized_cosine_from_lance(0.0), 0.0);
    }

    #[test]
    fn orthogonal_is_half() {
        // Lance: cosine distance between orthogonal vectors = 1.0
        let result = normalized_cosine_from_lance(1.0);
        assert!((result - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn opposite_is_one() {
        assert_eq!(normalized_cosine_from_lance(2.0), 1.0);
    }

    #[test]
    fn clamps_below_zero() {
        assert_eq!(normalized_cosine_from_lance(-0.5), 0.0);
    }

    #[test]
    fn clamps_above_two() {
        assert_eq!(normalized_cosine_from_lance(2.5), 1.0);
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = vec![1.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
        assert!((v[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn l2_normalize_scales_to_unit() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_unchanged() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cosine_distance_identical_is_zero() {
        let mut a = vec![1.0, 2.0, 3.0];
        l2_normalize(&mut a);
        let result = cosine_distance_normalized(&a, &a);
        assert!(result.abs() < 1e-6, "identical vectors should have distance ~0, got {}", result);
    }

    #[test]
    fn cosine_distance_orthogonal_is_half() {
        // Two orthogonal unit vectors.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = cosine_distance_normalized(&a, &b);
        assert!(
            (result - 0.5).abs() < 1e-6,
            "orthogonal vectors should have distance ~0.5, got {}",
            result
        );
    }

    #[test]
    fn cosine_distance_opposite_is_one() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let result = cosine_distance_normalized(&a, &b);
        assert!(
            (result - 1.0).abs() < 1e-6,
            "opposite vectors should have distance ~1.0, got {}",
            result
        );
    }

    #[test]
    fn cosine_distance_consistent_with_lance_normalization() {
        // Verify that cosine_distance_normalized gives the same result
        // as normalized_cosine_from_lance for a known dot product.
        let mut a = vec![3.0, 4.0, 0.0];
        let mut b = vec![4.0, 3.0, 0.0];
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let lance_dist = 1.0 - dot;
        let expected = normalized_cosine_from_lance(lance_dist);
        let actual = cosine_distance_normalized(&a, &b);
        assert!(
            (actual - expected).abs() < 1e-6,
            "cosine_distance_normalized should match normalized_cosine_from_lance: {} vs {}",
            actual,
            expected
        );
    }
}
