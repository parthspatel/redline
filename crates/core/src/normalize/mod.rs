//! Normalizer trait and normalization result types.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::char_mapping::CharMapping;
use crate::error::NormalizeError;

/// Result of a normalization operation.
#[derive(Debug, Clone)]
pub struct NormalizationResult {
    /// The normalized text.
    pub text: String,
    /// Mapping from original byte positions to normalized byte positions.
    pub mapping: CharMapping,
}

/// Trait for text normalizers. All normalizers are Send + Sync for thread safety.
pub trait Normalizer: Send + Sync {
    /// Apply normalization to input text.
    fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError>;

    /// Human-readable name used as the layer identifier.
    fn name(&self) -> &str;

    /// Estimated computational cost (0.0 = trivial, 1.0 = expensive).
    fn cost(&self) -> f32 {
        0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;

    assert_impl_all!(NormalizationResult: Send, Sync, Clone, core::fmt::Debug);

    struct DummyNormalizer;

    impl Normalizer for DummyNormalizer {
        fn normalize(&self, input: &str) -> Result<NormalizationResult, NormalizeError> {
            Ok(NormalizationResult {
                text: input.into(),
                mapping: CharMapping::identity(input.len() as u32)?,
            })
        }

        fn name(&self) -> &str {
            "dummy"
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn normalizer_is_send_sync() {
        assert_send_sync::<DummyNormalizer>();
    }

    #[test]
    fn normalizer_default_cost() {
        let n = DummyNormalizer;
        assert!((n.cost() - 0.5).abs() < f32::EPSILON);
    }
}
