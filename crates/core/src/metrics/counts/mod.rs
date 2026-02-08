//! Count-based metrics: word, char, byte, sentence, syllable, unique word,
//! paragraph, line, letter, digit, whitespace, punctuation.

pub mod byte_count;
pub mod char_count;
pub mod digit_count;
pub mod letter_count;
pub mod line_count;
pub mod paragraph_count;
pub mod punctuation_count;
pub mod sentence_count;
pub mod syllable_count;
pub mod unique_word_count;
pub mod whitespace_count;
pub mod word_count;

pub use byte_count::ByteCountMetric;
pub use char_count::CharCountMetric;
pub use digit_count::DigitCountMetric;
pub use letter_count::LetterCountMetric;
pub use line_count::LineCountMetric;
pub use paragraph_count::ParagraphCountMetric;
pub use punctuation_count::PunctuationCountMetric;
pub use sentence_count::SentenceCountMetric;
pub use syllable_count::{SyllableCountMetric, count_syllables};
pub use unique_word_count::UniqueWordCountMetric;
pub use whitespace_count::WhitespaceCountMetric;
pub use word_count::WordCountMetric;
