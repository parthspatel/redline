//! Tokenizer trait for text tokenization.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::char_mapping::CharMapping;
use crate::error::TokenizeError;
use crate::text_store::TextStoreBuilder;
use crate::token::Token;

/// Trait for tokenizers. All tokenizers are Send + Sync for thread safety.
pub trait Tokenizer: Send + Sync {
    /// Tokenize text, interning token strings into the store.
    fn tokenize(
        &self,
        text: &str,
        mapping: &CharMapping,
        store: &mut TextStoreBuilder,
    ) -> Result<Vec<Token>, TokenizeError>;

    /// Human-readable name of this tokenizer.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::Span;
    use crate::token::TokenKind;

    struct DummyTokenizer;

    impl Tokenizer for DummyTokenizer {
        fn tokenize(
            &self,
            text: &str,
            _mapping: &CharMapping,
            store: &mut TextStoreBuilder,
        ) -> Result<Vec<Token>, TokenizeError> {
            let id = store.intern(text);
            Ok(vec![Token::new(
                id,
                Span::new(0, text.len() as u32),
                TokenKind::Regular,
            )])
        }

        fn name(&self) -> &str {
            "dummy"
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn tokenizer_is_send_sync() {
        assert_send_sync::<DummyTokenizer>();
    }
}
