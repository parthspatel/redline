//! Text normalization pipeline
//!
//! Provides a layered approach to text normalization where each normalizer
//! creates a new layer with mappings back to the original text.

use crate::mapping::CharacterMap;
use crate::normalizers::Normalizer;

/// A single layer of normalized text with mapping to previous layer
#[derive(Debug, Clone)]
pub struct NormalizationLayer {
    /// The normalized text at this layer
    pub text: String,

    /// Mapping from this layer back to the previous layer
    /// For the first layer, this maps to the original input
    pub mapping: CharacterMap,

    /// Optional metadata about this normalization
    pub metadata: LayerMetadata,
}

/// Metadata about a normalization layer
#[derive(Debug, Clone, Default)]
pub struct LayerMetadata {
    /// Name of the normalizer that created this layer
    pub normalizer_name: String,

    /// Additional information (e.g., parameters used)
    pub info: Vec<(String, String)>,
}

impl NormalizationLayer {
    pub fn new(text: String, mapping: CharacterMap) -> Self {
        Self {
            text,
            mapping,
            metadata: LayerMetadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: LayerMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Map a position in this layer's text to the previous layer
    pub fn map_to_previous(&self, position: usize) -> Option<Vec<usize>> {
        self.mapping
            .normalized_to_original(position)
            .map(|m| m.all_positions())
    }

    /// Get the length of the text in this layer
    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// A pipeline of text normalizers that creates layered transformations
#[derive(Default, Clone)]
pub struct TextPipeline {
    /// Ordered list of normalizers to apply
    normalizers: Vec<Box<dyn Normalizer>>,
}

impl TextPipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Self {
            normalizers: Vec::new(),
        }
    }

    /// Add a normalizer to the pipeline
    pub fn add_normalizer(mut self, normalizer: Box<dyn Normalizer>) -> Self {
        self.normalizers.push(normalizer);
        self
    }

    /// Add multiple normalizers at once
    pub fn add_normalizers(mut self, normalizers: Vec<Box<dyn Normalizer>>) -> Self {
        self.normalizers.extend(normalizers);
        self
    }

    /// Process text through the entire pipeline, returning all layers
    ///
    /// # Arguments
    ///
    /// * `input` - The original input text
    ///
    /// # Returns
    ///
    /// A `LayerSet` containing all transformation layers including the original
    pub fn process(&self, input: &str) -> LayerSet {
        let mut layers = LayerSet::new(input.to_string());

        let mut current_text = input.to_string();

        for normalizer in &self.normalizers {
            let (normalized, mapping) = normalizer.normalize(&current_text);

            let metadata = LayerMetadata {
                normalizer_name: normalizer.name().to_string(),
                info: normalizer.metadata(),
            };

            let layer = NormalizationLayer {
                text: normalized.clone(),
                mapping,
                metadata,
            };

            layers.add_layer(layer);
            current_text = normalized;
        }

        layers
    }

    /// Get the number of normalizers in the pipeline
    pub fn len(&self) -> usize {
        self.normalizers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.normalizers.is_empty()
    }

    /// Get the names of all normalizers in the pipeline
    pub fn normalizer_names(&self) -> Vec<String> {
        self.normalizers
            .iter()
            .map(|n| n.name().to_string())
            .collect()
    }
}

/// A complete set of normalization layers from original to final
#[derive(Debug, Clone)]
pub struct LayerSet {
    /// The original input text (layer 0)
    original: String,

    /// All normalization layers (layer 1, 2, 3, ...)
    layers: Vec<NormalizationLayer>,
}

impl LayerSet {
    /// Create a new layer set with just the original text
    pub fn new(original: String) -> Self {
        Self {
            original,
            layers: Vec::new(),
        }
    }

    /// Add a new normalization layer
    pub fn add_layer(&mut self, layer: NormalizationLayer) {
        self.layers.push(layer);
    }

    /// Get the original text
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Get a specific layer by index (0 = original, 1 = first normalization, etc.)
    pub fn layer(&self, index: usize) -> Option<&str> {
        if index == 0 {
            Some(&self.original)
        } else {
            self.layers.get(index - 1).map(|l| l.text.as_str())
        }
    }

    /// Get the final (most normalized) layer
    pub fn final_layer(&self) -> &str {
        self.layers
            .last()
            .map(|l| l.text.as_str())
            .unwrap_or(&self.original)
    }

    /// Get the total number of layers (including original)
    pub fn num_layers(&self) -> usize {
        self.layers.len() + 1 // +1 for original
    }

    /// Get all layer texts as a vector
    pub fn all_texts(&self) -> Vec<&str> {
        let mut texts = vec![self.original.as_str()];
        texts.extend(self.layers.iter().map(|l| l.text.as_str()));
        texts
    }

    /// Map a position in a specific layer back to the original text
    ///
    /// # Arguments
    ///
    /// * `layer_index` - The layer to map from (0 = original, 1 = first norm, etc.)
    /// * `position` - Position in that layer
    ///
    /// # Returns
    ///
    /// Positions in the original text
    pub fn map_to_original(&self, layer_index: usize, position: usize) -> Vec<usize> {
        if layer_index == 0 {
            // Already in original
            return vec![position];
        }

        if layer_index > self.layers.len() {
            return vec![];
        }

        // Start from the specified layer and work backwards
        let mut current_positions = vec![position];

        for i in (0..layer_index).rev() {
            let layer = &self.layers[i];
            let mut next_positions = Vec::new();

            for &pos in &current_positions {
                if let Some(mapped) = layer.mapping.normalized_to_original(pos) {
                    next_positions.extend(mapped.all_positions());
                }
            }

            current_positions = next_positions;
        }

        current_positions
    }

    /// Get a composed mapping from the final layer directly to the original
    pub fn get_final_to_original_mapping(&self) -> CharacterMap {
        if self.layers.is_empty() {
            return CharacterMap::identity(self.original.len());
        }

        // Compose all mappings in reverse order
        let mut composed = self.layers[0].mapping.clone();

        for layer in &self.layers[1..] {
            composed = composed.compose(&layer.mapping);
        }

        composed
    }

    /// Get metadata for all layers
    pub fn layer_metadata(&self) -> Vec<&LayerMetadata> {
        self.layers.iter().map(|l| &l.metadata).collect()
    }

    /// Create an iterator over all layers (including original as layer 0)
    pub fn iter(&self) -> LayerIterator<'_> {
        LayerIterator {
            layer_set: self,
            current_index: 0,
        }
    }
}

/// Iterator over layers in a LayerSet
pub struct LayerIterator<'a> {
    layer_set: &'a LayerSet,
    current_index: usize,
}

impl<'a> Iterator for LayerIterator<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.layer_set.layer(self.current_index);
        if result.is_some() {
            self.current_index += 1;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::Lowercase;

    #[test]
    fn test_empty_pipeline() {
        let pipeline = TextPipeline::new();
        let layers = pipeline.process("Hello World");

        assert_eq!(layers.num_layers(), 1); // Just original
        assert_eq!(layers.original(), "Hello World");
        assert_eq!(layers.final_layer(), "Hello World");
    }

    #[test]
    fn test_single_normalizer() {
        let pipeline = TextPipeline::new().add_normalizer(Box::new(Lowercase));

        let layers = pipeline.process("Hello World");

        assert_eq!(layers.num_layers(), 2); // Original + 1 normalization
        assert_eq!(layers.original(), "Hello World");
        assert_eq!(layers.final_layer(), "hello world");
    }

    #[test]
    fn test_layer_iteration() {
        let pipeline = TextPipeline::new().add_normalizer(Box::new(Lowercase));

        let layers = pipeline.process("TEST");
        let texts: Vec<&str> = layers.iter().collect();

        assert_eq!(texts, vec!["TEST", "test"]);
    }

    #[test]
    fn test_map_to_original() {
        let pipeline = TextPipeline::new().add_normalizer(Box::new(Lowercase));

        let layers = pipeline.process("ABC");

        // Position 1 in layer 1 should map to position 1 in original
        let positions = layers.map_to_original(1, 1);
        assert_eq!(positions, vec![1]);
    }
}
