//! Text tokenizers
//!
//! Provides the `Tokenizer` trait and various implementations for splitting text
//! into tokens. Tokenizers work on normalized text and maintain mappings back
//! to the original.

use crate::mapping::CharSpan;
use crate::pipeline::LayerSet;

/// Represents a single token with metadata
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The token text (from the normalized layer)
    pub text: String,
    
    /// Position span in the normalized layer
    pub normalized_span: CharSpan,
    
    /// Position span(s) in the original text
    pub original_spans: Vec<CharSpan>,
    
    /// Token index in the sequence
    pub index: usize,
    
    /// Optional metadata
    pub metadata: TokenMetadata,
}

/// Metadata associated with a token
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenMetadata {
    /// Token type (e.g., "word", "punctuation", "whitespace")
    pub token_type: Option<String>,
    
    /// Additional properties
    pub properties: Vec<(String, String)>,
}

impl Token {
    /// Create a new token
    pub fn new(
        text: String,
        normalized_span: CharSpan,
        original_spans: Vec<CharSpan>,
        index: usize,
    ) -> Self {
        Self {
            text,
            normalized_span,
            original_spans,
            index,
            metadata: TokenMetadata::default(),
        }
    }

    /// Add metadata to this token
    pub fn with_metadata(mut self, metadata: TokenMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the start position in the normalized text
    pub fn normalized_start(&self) -> usize {
        self.normalized_span.start
    }

    /// Get the end position in the normalized text
    pub fn normalized_end(&self) -> usize {
        self.normalized_span.end
    }

    /// Get the start position in the original text
    pub fn original_start(&self) -> Option<usize> {
        self.original_spans.first().map(|s| s.start)
    }

    /// Get the end position in the original text
    pub fn original_end(&self) -> Option<usize> {
        self.original_spans.last().map(|s| s.end)
    }
}

/// Trait for tokenizers that split text into tokens
pub trait Tokenizer: Send + Sync {
    /// Tokenize the text from a layer set
    ///
    /// # Arguments
    ///
    /// * `layer_set` - The complete layer set with all normalizations
    ///
    /// # Returns
    ///
    /// A vector of tokens with mappings to original text
    fn tokenize(&self, layer_set: &LayerSet) -> Vec<Token>;

    /// Get the name of this tokenizer
    fn name(&self) -> &str;

    /// Get metadata about this tokenizer
    fn metadata(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Clone this tokenizer into a Box
    fn clone_box(&self) -> Box<dyn Tokenizer>;
}

// Implement Clone for Box<dyn Tokenizer>
impl Clone for Box<dyn Tokenizer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ============================================================================
// Built-in Tokenizers
// ============================================================================

/// Character-level tokenizer (splits into individual characters)
#[derive(Clone)]
pub struct CharacterTokenizer;

impl Tokenizer for CharacterTokenizer {
    fn tokenize(&self, layer_set: &LayerSet) -> Vec<Token> {
        let final_text = layer_set.final_layer();
        let final_layer_index = layer_set.num_layers() - 1;
        
        final_text
            .char_indices()
            .enumerate()
            .map(|(index, (pos, ch))| {
                let normalized_span = CharSpan::new(pos, pos + ch.len_utf8());
                let original_positions = layer_set.map_to_original(final_layer_index, pos);
                
                let original_spans = if original_positions.is_empty() {
                    vec![]
                } else {
                    vec![CharSpan::new(
                        *original_positions.first().unwrap(),
                        *original_positions.last().unwrap() + 1,
                    )]
                };

                Token::new(
                    ch.to_string(),
                    normalized_span,
                    original_spans,
                    index,
                )
            })
            .collect()
    }

    fn name(&self) -> &str {
        "character"
    }

    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

/// Word tokenizer (splits on whitespace and punctuation)
#[derive(Clone)]
pub struct WordTokenizer {
    /// Whether to include punctuation as separate tokens
    pub include_punctuation: bool,
    /// Whether to include whitespace as separate tokens
    pub include_whitespace: bool,
}

impl WordTokenizer {
    pub fn new() -> Self {
        Self {
            include_punctuation: false,
            include_whitespace: false,
        }
    }

    pub fn with_punctuation(mut self, include: bool) -> Self {
        self.include_punctuation = include;
        self
    }

    pub fn with_whitespace(mut self, include: bool) -> Self {
        self.include_whitespace = include;
        self
    }
}

impl Default for WordTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize(&self, layer_set: &LayerSet) -> Vec<Token> {
        let final_text = layer_set.final_layer();
        let final_layer_index = layer_set.num_layers() - 1;
        
        let mut tokens = Vec::new();
        let mut token_index = 0;
        let mut current_start = 0;
        let mut current_type: Option<TokenType> = None;

        for (pos, ch) in final_text.char_indices() {
            let char_type = classify_char(ch);

            match (&current_type, char_type) {
                (None, _) => {
                    // Start new token
                    current_start = pos;
                    current_type = Some(char_type);
                }
                (Some(prev_type), current) if *prev_type == current => {
                    // Continue current token
                }
                (Some(prev_type), current) => {
                    // Emit previous token
                    if should_include_token(*prev_type, self.include_punctuation, self.include_whitespace) {
                        let token = create_token(
                            final_text,
                            current_start,
                            pos,
                            token_index,
                            final_layer_index,
                            layer_set,
                            *prev_type,
                        );
                        tokens.push(token);
                        token_index += 1;
                    }

                    // Start new token
                    current_start = pos;
                    current_type = Some(current);
                }
            }
        }

        // Emit final token
        if let Some(token_type) = current_type {
            if should_include_token(token_type, self.include_punctuation, self.include_whitespace) {
                let token = create_token(
                    final_text,
                    current_start,
                    final_text.len(),
                    token_index,
                    final_layer_index,
                    layer_set,
                    token_type,
                );
                tokens.push(token);
            }
        }

        tokens
    }

    fn name(&self) -> &str {
        "word"
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("include_punctuation".to_string(), self.include_punctuation.to_string()),
            ("include_whitespace".to_string(), self.include_whitespace.to_string()),
        ]
    }

    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenType {
    Word,
    Punctuation,
    Whitespace,
    Number,
}

fn classify_char(ch: char) -> TokenType {
    if ch.is_whitespace() {
        TokenType::Whitespace
    } else if ch.is_ascii_punctuation() {
        TokenType::Punctuation
    } else if ch.is_numeric() {
        TokenType::Number
    } else {
        TokenType::Word
    }
}

fn should_include_token(token_type: TokenType, include_punct: bool, include_ws: bool) -> bool {
    match token_type {
        TokenType::Word | TokenType::Number => true,
        TokenType::Punctuation => include_punct,
        TokenType::Whitespace => include_ws,
    }
}

fn create_token(
    text: &str,
    start: usize,
    end: usize,
    index: usize,
    layer_index: usize,
    layer_set: &LayerSet,
    token_type: TokenType,
) -> Token {
    let token_text = text[start..end].to_string();
    let normalized_span = CharSpan::new(start, end);
    
    // Map all positions in the token back to original
    let mut original_positions = Vec::new();
    for pos in start..end {
        original_positions.extend(layer_set.map_to_original(layer_index, pos));
    }
    
    let original_spans = if original_positions.is_empty() {
        vec![]
    } else {
        original_positions.sort_unstable();
        original_positions.dedup();
        
        // Group consecutive positions into spans
        let mut spans = Vec::new();
        let mut span_start = original_positions[0];
        let mut span_end = span_start + 1;
        
        for &pos in &original_positions[1..] {
            if pos == span_end {
                span_end = pos + 1;
            } else {
                spans.push(CharSpan::new(span_start, span_end));
                span_start = pos;
                span_end = pos + 1;
            }
        }
        spans.push(CharSpan::new(span_start, span_end));
        
        spans
    };

    let mut metadata = TokenMetadata::default();
    metadata.token_type = Some(format!("{:?}", token_type).to_lowercase());

    Token::new(token_text, normalized_span, original_spans, index)
        .with_metadata(metadata)
}

/// N-gram tokenizer (creates overlapping n-character sequences)
#[derive(Clone)]
pub struct NGramTokenizer {
    /// Size of each n-gram
    pub n: usize,
    /// Whether to use character-level or word-level n-grams
    pub character_level: bool,
}

impl NGramTokenizer {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            character_level: true,
        }
    }

    pub fn character_level(mut self) -> Self {
        self.character_level = true;
        self
    }

    pub fn word_level(mut self) -> Self {
        self.character_level = false;
        self
    }
}

impl Tokenizer for NGramTokenizer {
    fn tokenize(&self, layer_set: &LayerSet) -> Vec<Token> {
        if self.character_level {
            self.tokenize_char_ngrams(layer_set)
        } else {
            self.tokenize_word_ngrams(layer_set)
        }
    }

    fn name(&self) -> &str {
        if self.character_level {
            "char_ngram"
        } else {
            "word_ngram"
        }
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("n".to_string(), self.n.to_string()),
            ("character_level".to_string(), self.character_level.to_string()),
        ]
    }

    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

impl NGramTokenizer {
    fn tokenize_char_ngrams(&self, layer_set: &LayerSet) -> Vec<Token> {
        let final_text = layer_set.final_layer();
        let final_layer_index = layer_set.num_layers() - 1;
        let chars: Vec<(usize, char)> = final_text.char_indices().collect();

        if chars.len() < self.n {
            return vec![];
        }

        (0..=chars.len() - self.n)
            .enumerate()
            .map(|(index, i)| {
                let start_pos = chars[i].0;
                let end_pos = if i + self.n < chars.len() {
                    chars[i + self.n].0
                } else {
                    final_text.len()
                };

                let token_text = final_text[start_pos..end_pos].to_string();
                let normalized_span = CharSpan::new(start_pos, end_pos);

                // Map to original
                let mut original_positions = Vec::new();
                for pos in start_pos..end_pos {
                    original_positions.extend(layer_set.map_to_original(final_layer_index, pos));
                }

                let original_spans = if original_positions.is_empty() {
                    vec![]
                } else {
                    vec![CharSpan::new(
                        *original_positions.first().unwrap(),
                        *original_positions.last().unwrap() + 1,
                    )]
                };

                Token::new(token_text, normalized_span, original_spans, index)
            })
            .collect()
    }

    fn tokenize_word_ngrams(&self, layer_set: &LayerSet) -> Vec<Token> {
        // First tokenize into words
        let word_tokenizer = WordTokenizer::new();
        let words = word_tokenizer.tokenize(layer_set);

        if words.len() < self.n {
            return vec![];
        }

        (0..=words.len() - self.n)
            .enumerate()
            .map(|(index, i)| {
                let ngram_words = &words[i..i + self.n];
                let token_text = ngram_words
                    .iter()
                    .map(|w| w.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let start = ngram_words.first().unwrap().normalized_span.start;
                let end = ngram_words.last().unwrap().normalized_span.end;
                let normalized_span = CharSpan::new(start, end);

                let original_spans = ngram_words
                    .iter()
                    .flat_map(|w| w.original_spans.clone())
                    .collect();

                Token::new(token_text, normalized_span, original_spans, index)
            })
            .collect()
    }
}

/// Sentence tokenizer (splits on sentence boundaries)
#[derive(Clone)]
pub struct SentenceTokenizer;

impl Tokenizer for SentenceTokenizer {
    fn tokenize(&self, layer_set: &LayerSet) -> Vec<Token> {
        let final_text = layer_set.final_layer();
        let final_layer_index = layer_set.num_layers() - 1;

        // Simple sentence boundary detection
        let mut tokens = Vec::new();
        let mut token_index = 0;
        let mut sent_start = 0;

        for (pos, ch) in final_text.char_indices() {
            if matches!(ch, '.' | '!' | '?') {
                let end = pos + ch.len_utf8();
                
                // Skip ahead past any trailing whitespace
                let final_end = final_text[end..]
                    .char_indices()
                    .take_while(|(_, c)| c.is_whitespace())
                    .last()
                    .map(|(i, c)| end + i + c.len_utf8())
                    .unwrap_or(end);

                if sent_start < final_end {
                    let token = create_sentence_token(
                        final_text,
                        sent_start,
                        final_end,
                        token_index,
                        final_layer_index,
                        layer_set,
                    );
                    tokens.push(token);
                    token_index += 1;
                    sent_start = final_end;
                }
            }
        }

        // Add final sentence if there's remaining text
        if sent_start < final_text.len() {
            let token = create_sentence_token(
                final_text,
                sent_start,
                final_text.len(),
                token_index,
                final_layer_index,
                layer_set,
            );
            tokens.push(token);
        }

        tokens
    }

    fn name(&self) -> &str {
        "sentence"
    }

    fn clone_box(&self) -> Box<dyn Tokenizer> {
        Box::new(self.clone())
    }
}

fn create_sentence_token(
    text: &str,
    start: usize,
    end: usize,
    index: usize,
    layer_index: usize,
    layer_set: &LayerSet,
) -> Token {
    let token_text = text[start..end].trim().to_string();
    let normalized_span = CharSpan::new(start, end);
    
    let mut original_positions = Vec::new();
    for pos in start..end {
        original_positions.extend(layer_set.map_to_original(layer_index, pos));
    }
    
    let original_spans = if original_positions.is_empty() {
        vec![]
    } else {
        vec![CharSpan::new(
            *original_positions.first().unwrap(),
            *original_positions.last().unwrap() + 1,
        )]
    };

    Token::new(token_text, normalized_span, original_spans, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::TextPipeline;

    #[test]
    fn test_character_tokenizer() {
        let pipeline = TextPipeline::new();
        let layers = pipeline.process("abc");
        let tokenizer = CharacterTokenizer;
        let tokens = tokenizer.tokenize(&layers);
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "a");
        assert_eq!(tokens[1].text, "b");
        assert_eq!(tokens[2].text, "c");
    }

    #[test]
    fn test_word_tokenizer() {
        let pipeline = TextPipeline::new();
        let layers = pipeline.process("hello world");
        let tokenizer = WordTokenizer::new();
        let tokens = tokenizer.tokenize(&layers);
        
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_ngram_tokenizer() {
        let pipeline = TextPipeline::new();
        let layers = pipeline.process("abc");
        let tokenizer = NGramTokenizer::new(2);
        let tokens = tokenizer.tokenize(&layers);
        
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "ab");
        assert_eq!(tokens[1].text, "bc");
    }
}
