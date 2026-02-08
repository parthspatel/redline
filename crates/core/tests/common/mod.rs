// All text fixtures are original prose written for this project.
// No copyrighted material is included.

#![allow(dead_code)]

use std::sync::Arc;

use redline_core::process::ProcessedText;
use redline_core::tokenize::WordTokenizer;
use redline_core::{ExecutionMode, TextProcessor};

// ---------------------------------------------------------------------------
// Section 1: Shared helper functions
// ---------------------------------------------------------------------------

/// Process a string with no normalizers, WordTokenizer, ExecutionMode::All.
/// Replaces 30+ duplicated `process()` helpers across test files.
pub fn process(s: &str) -> ProcessedText {
    TextProcessor::new(vec![], Box::new(WordTokenizer), ExecutionMode::All)
        .unwrap()
        .process(s)
        .unwrap()
}

/// Wraps `process()` result in `Arc` for diff/metrics APIs that require shared ownership.
pub fn process_arc(s: &str) -> Arc<ProcessedText> {
    Arc::new(process(s))
}

// ---------------------------------------------------------------------------
// Section 2: Tiered English text constants by readability level
// ---------------------------------------------------------------------------

/// Grade level ~3, Flesch 90+. Short sentences, monosyllabic words.
pub const SIMPLE_PROSE: &str = "\
The cat sat on the warm mat. A small bird sang in the old oak tree. \
The sun was bright and the sky was clear. It was a good day to play.";

/// Grade level ~8, Flesch 60-70. Mixed sentence lengths, some polysyllabic words.
pub const LITERARY_PROSE: &str = "\
The afternoon light filtered through the dusty windows of the abandoned library. \
Volumes of forgotten stories lined the shelves, their spines cracked and faded. \
She traced her fingers along the titles, searching for something that might still hold meaning.";

/// Grade level ~12, Flesch 30-50. Domain terminology, longer sentences.
pub const TECHNICAL_PROSE: &str = "\
The algorithm employs a dynamic programming approach to compute the minimum edit distance \
between two sequences. Each cell in the matrix represents the optimal alignment cost, \
considering insertions, deletions, and substitutions. This computational framework enables \
efficient comparison of arbitrarily long input strings.";

/// Grade level ~16, Flesch <30. Polysyllabic jargon (40%+ words with 3+ syllables),
/// complex syntax. Designed for Gunning Fog and SMOG differentiation.
pub const ACADEMIC_PROSE: &str = "\
The epistemological implications of computational linguistics necessitate rigorous \
examination of morphosyntactic representations within probabilistic frameworks. \
Disambiguation of polysemous lexical items requires sophisticated contextual analysis \
methodologies incorporating distributional semantics and hierarchical abstraction.";

// ---------------------------------------------------------------------------
// Section 3: Multi-paragraph text
// ---------------------------------------------------------------------------

/// 2 paragraphs separated by `\n\n`, approximately 100 words total.
/// Good for paragraph_count and line_count metrics.
pub const MULTI_PARAGRAPH: &str = "\
The river wound through the valley like a silver thread. Tall grasses \
bent in the evening breeze, and the air carried the faint scent of \
wildflowers. A heron stood motionless at the water's edge, watching \
for the glint of fish beneath the surface.

Beyond the ridge, the village lights began to flicker on one by one. \
Smoke curled from chimneys into the darkening sky. The baker was the \
last to close his shop, sliding the iron bolt across the heavy oak \
door before walking home along the cobblestone lane.";

// ---------------------------------------------------------------------------
// Section 4: Realistic diff pairs
// ---------------------------------------------------------------------------

pub mod diff_pairs {
    //! Before/after text pairs for testing diff operations.

    /// Typo: "recieve" corrected to "receive". Should produce exactly 1 Replace
    /// operation and high similarity_ratio (>0.8).
    pub const TYPO_BEFORE: &str =
        "The committee agreed to recieve the annual budget report on Friday.";
    pub const TYPO_AFTER: &str =
        "The committee agreed to receive the annual budget report on Friday.";

    /// Insertion: a third sentence inserted between the original two.
    /// Should produce Insert operations with zero Delete operations.
    pub const INSERT_BEFORE: &str = "\
The project deadline was moved to next quarter. \
All team members should update their schedules accordingly.";
    pub const INSERT_AFTER: &str = "\
The project deadline was moved to next quarter. \
Management approved additional funding for the extended timeline. \
All team members should update their schedules accordingly.";

    /// Rewrite: same topic, substantially different vocabulary and structure.
    /// Should produce significant Replace/Delete/Insert operations and lower
    /// similarity_ratio.
    pub const REWRITE_BEFORE: &str = "\
The server experienced intermittent failures throughout the weekend. \
Engineers identified a memory leak in the connection pooling module.";
    pub const REWRITE_AFTER: &str = "\
Recurring outages plagued the production infrastructure over the past two days. \
Root cause analysis revealed that unbounded resource allocation in the networking \
layer was exhausting available system memory.";
}

// ---------------------------------------------------------------------------
// Section 5: Multi-language constants
// ---------------------------------------------------------------------------

pub mod multilang {
    //! Multi-language and multi-script text constants for internationalization testing.

    /// Plain ASCII text with mixed case and numbers.
    pub const ASCII: &str = "Hello  WORLD, this is a test! 123 numbers here.";

    /// French/Latin accented characters.
    pub const UNICODE_ACCENTED: &str = "Caf\u{00E9} r\u{00E9}sum\u{00E9} na\u{00EF}ve";

    /// Chinese + English mixed script.
    pub const CJK: &str = "\u{4F60}\u{597D}\u{4E16}\u{754C} Hello World";

    /// Emoji interspersed with Latin text.
    pub const EMOJI: &str = "Hello \u{1F44B} World \u{1F30D}";

    /// Arabic right-to-left text.
    pub const RTL: &str = "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627} \u{0628}\u{0627}\u{0644}\u{0639}\u{0627}\u{0644}\u{0645}";

    /// Vietnamese with combining diacritics.
    pub const VIETNAMESE: &str = "Vi\u{1EC7}t Nam \u{0111}\u{1EB9}p l\u{1EAF}m";

    /// Mixed script: Latin, CJK, Arabic, Emoji, accented.
    pub const MIXED_SCRIPT: &str =
        "Hello \u{4E16}\u{754C} \u{0645}\u{0631}\u{062D}\u{0628}\u{0627} \u{1F44B} caf\u{00E9}";

    /// Chinese paragraph: 3 short sentences with Chinese period.
    pub const CHINESE_PARAGRAPH: &str = "\
\u{6625}\u{5929}\u{7684}\u{82B1}\u{5F00}\u{4E86}\u{3002}\
\u{5C0F}\u{9E1F}\u{5728}\u{6811}\u{4E0A}\u{5531}\u{6B4C}\u{3002}\
\u{5B69}\u{5B50}\u{4EEC}\u{5728}\u{516C}\u{56ED}\u{91CC}\u{73A9}\u{8006}\u{3002}";

    /// Japanese mixed script: hiragana, katakana, kanji with Japanese punctuation.
    pub const JAPANESE_MIXED: &str = "\
\u{6771}\u{4EAC}\u{306F}\u{7F8E}\u{3057}\u{3044}\u{90FD}\u{5E02}\u{3067}\u{3059}\u{3002}\
\u{30B5}\u{30AF}\u{30E9}\u{306E}\u{82B1}\u{304C}\u{54B2}\u{304D}\u{307E}\u{3057}\u{305F}\u{3002}\
\u{96E8}\u{306E}\u{65E5}\u{306B}\u{30E9}\u{30FC}\u{30E1}\u{30F3}\u{3092}\u{98DF}\u{3079}\u{307E}\u{3057}\u{305F}\u{3002}";

    /// Arabic paragraph: 3 short sentences.
    pub const ARABIC_PARAGRAPH: &str = "\
\u{0627}\u{0644}\u{0634}\u{0645}\u{0633} \u{0645}\u{0634}\u{0631}\u{0642}\u{0629} \u{0627}\u{0644}\u{064A}\u{0648}\u{0645}. \
\u{0627}\u{0644}\u{0623}\u{0637}\u{0641}\u{0627}\u{0644} \u{064A}\u{0644}\u{0639}\u{0628}\u{0648}\u{0646} \u{0641}\u{064A} \u{0627}\u{0644}\u{062D}\u{062F}\u{064A}\u{0642}\u{0629}. \
\u{0627}\u{0644}\u{0637}\u{064A}\u{0648}\u{0631} \u{062A}\u{063A}\u{0631}\u{062F} \u{0641}\u{064A} \u{0627}\u{0644}\u{0633}\u{0645}\u{0627}\u{0621}.";

    /// French with accents, cedilla, and ligatures.
    pub const ACCENTED_FRENCH: &str = "\
Les gar\u{00E7}ons fran\u{00E7}ais pr\u{00E9}f\u{00E8}rent les cr\u{00EA}pes au caf\u{00E9}. \
L'\u{0153}uvre \u{00E9}tait magnifique, \u{00E0} c\u{00F4}t\u{00E9} du ch\u{00E2}teau.";
}
