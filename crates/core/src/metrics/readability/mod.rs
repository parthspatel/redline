//! Readability and complexity metrics.
//!
//! Batch 1 (plan 04-04): avg_word_length, avg_sentence_length, vocabulary_richness,
//! lexical_density, flesch_reading_ease.
//!
//! Batch 2 (plan 04-05): flesch_kincaid_grade, gunning_fog, smog_index,
//! coleman_liau, ari.

pub mod ari;
pub mod avg_sentence_length;
pub mod avg_word_length;
pub mod coleman_liau;
pub mod flesch_kincaid_grade;
pub mod flesch_reading_ease;
pub mod gunning_fog;
pub mod lexical_density;
pub mod smog_index;
pub mod vocabulary_richness;

pub use ari::AriMetric;
pub use avg_sentence_length::AvgSentenceLengthMetric;
pub use avg_word_length::AvgWordLengthMetric;
pub use coleman_liau::ColemanLiauMetric;
pub use flesch_kincaid_grade::FleschKincaidGradeMetric;
pub use flesch_reading_ease::FleschReadingEaseMetric;
pub use gunning_fog::GunningFogMetric;
pub use lexical_density::LexicalDensityMetric;
pub use smog_index::SmogIndexMetric;
pub use vocabulary_richness::VocabularyRichnessMetric;
