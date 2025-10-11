//! Text normalizers
//!
//! Provides the `Normalizer` trait and various implementations for text transformation.
//! Each normalizer transforms text while maintaining character-level mappings.

use crate::mapping::{CharacterMap, CharSpan};

/// Trait for text normalizers
pub trait Normalizer: Send + Sync {
    /// Normalize the input text
    ///
    /// # Arguments
    ///
    /// * `input` - The text to normalize
    ///
    /// # Returns
    ///
    /// A tuple of (normalized_text, character_map)
    fn normalize(&self, input: &str) -> (String, CharacterMap);

    /// Get the name of this normalizer
    fn name(&self) -> &str;

    /// Get metadata about this normalizer (e.g., configuration)
    fn metadata(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Clone this normalizer into a Box
    fn clone_box(&self) -> Box<dyn Normalizer>;
}

// Implement Clone for Box<dyn Normalizer>
impl Clone for Box<dyn Normalizer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// ============================================================================
// Built-in Normalizers
// ============================================================================

/// Converts all text to lowercase
#[derive(Clone)]
pub struct Lowercase;

impl Normalizer for Lowercase {
    fn normalize(&self, input: &str) -> (String, CharacterMap) {
        let normalized = input.to_lowercase();
        let mut map = CharacterMap::new(input.len(), normalized.len());

        // Handle character mapping
        let mut norm_pos = 0;
        for (orig_pos, ch) in input.char_indices() {
            let lowercased = ch.to_lowercase().to_string();
            let char_len = lowercased.len();

            if char_len == 1 {
                // Simple case: one-to-one
                map.add_direct_mapping(norm_pos, orig_pos);
                norm_pos += 1;
            } else {
                // Some characters expand when lowercased (e.g., İ -> i̇)
                let positions: Vec<usize> = (norm_pos..norm_pos + char_len).collect();
                map.add_expanded_mapping(positions, orig_pos);
                norm_pos += char_len;
            }
        }

        (normalized, map)
    }

    fn name(&self) -> &str {
        "lowercase"
    }

    fn clone_box(&self) -> Box<dyn Normalizer> {
        Box::new(self.clone())
    }
}

/// Converts all text to uppercase
#[derive(Clone)]
pub struct Uppercase;

impl Normalizer for Uppercase {
    fn normalize(&self, input: &str) -> (String, CharacterMap) {
        let normalized = input.to_uppercase();
        let mut map = CharacterMap::new(input.len(), normalized.len());

        let mut norm_pos = 0;
        for (orig_pos, ch) in input.char_indices() {
            let uppercased = ch.to_uppercase().to_string();
            let char_len = uppercased.len();

            if char_len == 1 {
                map.add_direct_mapping(norm_pos, orig_pos);
                norm_pos += 1;
            } else {
                let positions: Vec<usize> = (norm_pos..norm_pos + char_len).collect();
                map.add_expanded_mapping(positions, orig_pos);
                norm_pos += char_len;
            }
        }

        (normalized, map)
    }

    fn name(&self) -> &str {
        "uppercase"
    }

    fn clone_box(&self) -> Box<dyn Normalizer> {
        Box::new(self.clone())
    }
}

/// Normalizes whitespace: collapses multiple spaces, converts tabs/newlines to spaces
#[derive(Clone)]
pub struct WhitespaceNormalizer {
    /// Whether to collapse multiple consecutive whitespaces into one
    pub collapse: bool,
    /// Whether to trim leading/trailing whitespace
    pub trim: bool,
}

impl Default for WhitespaceNormalizer {
    fn default() -> Self {
        Self {
            collapse: true,
            trim: false,
        }
    }
}

impl WhitespaceNormalizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_collapse(mut self, collapse: bool) -> Self {
        self.collapse = collapse;
        self
    }

    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }
}

impl Normalizer for WhitespaceNormalizer {
    fn normalize(&self, input: &str) -> (String, CharacterMap) {
        let mut normalized = String::new();
        let mut map = CharacterMap::new(input.len(), 0); // Will update length later

        let mut norm_pos = 0;
        let mut in_whitespace = false;
        let mut whitespace_start = 0;

        let chars: Vec<(usize, char)> = input.char_indices().collect();
        let mut i = 0;

        // Handle leading trim
        if self.trim {
            while i < chars.len() && chars[i].1.is_whitespace() {
                map.add_deletion(chars[i].0);
                i += 1;
            }
        }

        while i < chars.len() {
            let (orig_pos, ch) = chars[i];

            if ch.is_whitespace() {
                if !in_whitespace {
                    in_whitespace = true;
                    whitespace_start = orig_pos;
                }
                i += 1;
            } else {
                // Emit collapsed whitespace if needed
                if in_whitespace {
                    let whitespace_end = orig_pos;
                    
                    if self.collapse {
                        normalized.push(' ');
                        map.add_collapsed_mapping(
                            norm_pos,
                            CharSpan::new(whitespace_start, whitespace_end),
                        );
                        norm_pos += 1;
                    } else {
                        // Keep all whitespace
                        for j in whitespace_start..whitespace_end {
                            normalized.push(' ');
                            map.add_direct_mapping(norm_pos, j);
                            norm_pos += 1;
                        }
                    }
                    
                    in_whitespace = false;
                }

                // Emit the character
                normalized.push(ch);
                map.add_direct_mapping(norm_pos, orig_pos);
                norm_pos += 1;
                i += 1;
            }
        }

        // Handle trailing whitespace
        if in_whitespace && !self.trim {
            let whitespace_end = input.len();
            if self.collapse {
                normalized.push(' ');
                map.add_collapsed_mapping(
                    norm_pos,
                    CharSpan::new(whitespace_start, whitespace_end),
                );
            }
        }

        // Update the normalized length
        // let final_map = CharacterMap::new(input.len(), normalized.len());
        (normalized, map)
    }

    fn name(&self) -> &str {
        "whitespace"
    }

    fn metadata(&self) -> Vec<(String, String)> {
        vec![
            ("collapse".to_string(), self.collapse.to_string()),
            ("trim".to_string(), self.trim.to_string()),
        ]
    }

    fn clone_box(&self) -> Box<dyn Normalizer> {
        Box::new(self.clone())
    }
}

/// Removes all punctuation
#[derive(Clone)]
pub struct RemovePunctuation;

impl Normalizer for RemovePunctuation {
    fn normalize(&self, input: &str) -> (String, CharacterMap) {
        let mut normalized = String::new();
        let mut map = CharacterMap::new(input.len(), 0);
        let mut norm_pos = 0;

        for (orig_pos, ch) in input.char_indices() {
            if !ch.is_ascii_punctuation() {
                normalized.push(ch);
                map.add_direct_mapping(norm_pos, orig_pos);
                norm_pos += 1;
            } else {
                map.add_deletion(orig_pos);
            }
        }

        // let final_map = CharacterMap::new(input.len(), normalized.len());
        (normalized, map)
    }

    fn name(&self) -> &str {
        "remove_punctuation"
    }

    fn clone_box(&self) -> Box<dyn Normalizer> {
        Box::new(self.clone())
    }
}

/// Removes accents/diacritics (e.g., é -> e)
#[derive(Clone)]
pub struct RemoveAccents;

impl Normalizer for RemoveAccents {
    fn normalize(&self, input: &str) -> (String, CharacterMap) {
        // Use Unicode normalization to decompose characters
        use unicode_normalization::UnicodeNormalization;
        
        let mut normalized = String::new();
        let mut map = CharacterMap::new(input.len(), 0);
        let mut norm_pos = 0;

        for (orig_pos, ch) in input.char_indices() {
            let decomposed: String = ch.nfd().collect();
            
            // Keep only the base character (first in decomposition)
            if let Some(base_char) = decomposed.chars().next() {
                if !base_char.is_ascii_punctuation() 
                    && !is_combining_mark(base_char) {
                    normalized.push(base_char);
                    map.add_direct_mapping(norm_pos, orig_pos);
                    norm_pos += 1;
                }
            }
        }

        // let final_map = CharacterMap::new(input.len(), normalized.len());
        (normalized, map)
    }

    fn name(&self) -> &str {
        "remove_accents"
    }

    fn clone_box(&self) -> Box<dyn Normalizer> {
        Box::new(self.clone())
    }
}

fn is_combining_mark(ch: char) -> bool {
    matches!(ch, '\u{0300}'..='\u{036F}' | '\u{1AB0}'..='\u{1AFF}' | '\u{1DC0}'..='\u{1DFF}' | '\u{20D0}'..='\u{20FF}' | '\u{FE20}'..='\u{FE2F}')
}

/// Removes numbers
#[derive(Clone)]
pub struct RemoveNumbers;

impl Normalizer for RemoveNumbers {
    fn normalize(&self, input: &str) -> (String, CharacterMap) {
        let mut normalized = String::new();
        let mut map = CharacterMap::new(input.len(), 0);
        let mut norm_pos = 0;

        for (orig_pos, ch) in input.char_indices() {
            if !ch.is_numeric() {
                normalized.push(ch);
                map.add_direct_mapping(norm_pos, orig_pos);
                norm_pos += 1;
            } else {
                map.add_deletion(orig_pos);
            }
        }

        // let final_map = CharacterMap::new(input.len(), normalized.len());
        (normalized, map)
    }

    fn name(&self) -> &str {
        "remove_numbers"
    }

    fn clone_box(&self) -> Box<dyn Normalizer> {
        Box::new(self.clone())
    }
}

/// Custom normalizer using a user-provided function
pub struct CustomNormalizer {
    name: String,
    func: Box<dyn Fn(&str) -> String + Send + Sync>,
}

impl CustomNormalizer {
    pub fn new<F>(name: impl Into<String>, func: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            func: Box::new(func),
        }
    }
}

impl Normalizer for CustomNormalizer {
    fn normalize(&self, input: &str) -> (String, CharacterMap) {
        let normalized = (self.func)(input);
        
        // Build a simple character-by-character mapping
        // This is a simplification - custom normalizers should ideally
        // provide their own mapping logic
        let mut map = CharacterMap::new(input.len(), normalized.len());
        
        let min_len = input.len().min(normalized.len());
        for i in 0..min_len {
            map.add_direct_mapping(i, i);
        }
        
        // Handle length differences
        if normalized.len() > input.len() {
            for i in input.len()..normalized.len() {
                map.add_insertion(i);
            }
        } else if input.len() > normalized.len() {
            for i in normalized.len()..input.len() {
                map.add_deletion(i);
            }
        }

        (normalized, map)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn clone_box(&self) -> Box<dyn Normalizer> {
        panic!("CustomNormalizer cannot be cloned");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lowercase() {
        let normalizer = Lowercase;
        let (result, _map) = normalizer.normalize("Hello World");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_whitespace_collapse() {
        let normalizer = WhitespaceNormalizer::new().with_collapse(true);
        let (result, _map) = normalizer.normalize("Hello    World");
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_remove_punctuation() {
        let normalizer = RemovePunctuation;
        let (result, _map) = normalizer.normalize("Hello, World!");
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_remove_numbers() {
        let normalizer = RemoveNumbers;
        let (result, _map) = normalizer.normalize("Test123");
        assert_eq!(result, "Test");
    }
}
