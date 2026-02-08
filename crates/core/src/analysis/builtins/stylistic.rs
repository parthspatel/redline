//! StylisticAnalyzer: voice shifts, tone shifts, structural style changes.

use core::any::Any;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, string::ToString, vec, vec::Vec};

use crate::analysis::{
    AnalysisContext, AnalysisError, AnalysisReport, AnalyzerDependency, AnalyzerMeta,
    AnalyzerPlugin,
};
use crate::diff::edit_operation::EditKind;

/// Detected voice type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Voice {
    Active,
    Passive,
    Unknown,
}

/// A detected voice shift between source and target.
#[derive(Debug, Clone)]
pub struct VoiceShift {
    pub operation_index: usize,
    pub from: Voice,
    pub to: Voice,
    pub confidence: f64,
}

/// Tone measurement dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneDimension {
    FormalInformal,
    HedgingAssertive,
}

/// A detected tone shift.
#[derive(Debug, Clone)]
pub struct ToneShift {
    pub operation_index: usize,
    pub dimension: ToneDimension,
    pub from_score: f64,
    pub to_score: f64,
    pub confidence: f64,
}

/// Structural style measurements.
#[derive(Debug, Clone)]
pub struct StructuralChanges {
    pub source_avg_sentence_length: f64,
    pub target_avg_sentence_length: f64,
    pub sentence_length_delta: f64,
    pub source_avg_word_length: f64,
    pub target_avg_word_length: f64,
    pub word_length_delta: f64,
}

/// Result of stylistic analysis.
#[derive(Debug, Clone)]
pub struct StylisticResult {
    pub voice_shifts: Vec<VoiceShift>,
    pub tone_shifts: Vec<ToneShift>,
    pub structural_changes: StructuralChanges,
}

// ── Heuristic helpers ─────────────────────────────────────────

const TO_BE_FORMS: &[&str] = &["am", "is", "are", "was", "were", "be", "been", "being"];

const IRREGULAR_PAST_PARTICIPLES: &[&str] = &[
    "made",
    "done",
    "gone",
    "said",
    "told",
    "shown",
    "known",
    "given",
    "taken",
    "seen",
    "found",
    "thought",
    "brought",
    "bought",
    "caught",
    "taught",
    "felt",
    "left",
    "held",
    "kept",
    "meant",
    "met",
    "paid",
    "put",
    "run",
    "read",
    "sent",
    "set",
    "sat",
    "spent",
    "stood",
    "lost",
    "cut",
    "built",
    "led",
    "understood",
    "written",
    "broken",
    "chosen",
    "driven",
    "eaten",
    "fallen",
    "forgotten",
    "frozen",
    "hidden",
    "ridden",
    "spoken",
    "stolen",
    "sworn",
    "thrown",
    "worn",
];

fn is_past_participle(word: &str) -> bool {
    let lower = word.to_lowercase();
    if IRREGULAR_PAST_PARTICIPLES.contains(&lower.as_str()) {
        return true;
    }
    lower.ends_with("ed") || lower.ends_with("en") || lower.ends_with("wn") || lower.ends_with("ne")
}

fn passive_voice_ratio(words: &[&str]) -> f64 {
    if words.len() < 2 {
        return 0.0;
    }
    let mut passive_count = 0usize;
    for window in words.windows(2) {
        let w0 = window[0].to_lowercase();
        if TO_BE_FORMS.contains(&w0.as_str()) && is_past_participle(window[1]) {
            passive_count += 1;
        }
    }
    passive_count as f64 / (words.len() - 1).max(1) as f64
}

fn classify_voice(words: &[&str]) -> Voice {
    let ratio = passive_voice_ratio(words);
    if ratio > 0.3 {
        Voice::Passive
    } else if ratio < 0.1 {
        Voice::Active
    } else {
        Voice::Unknown
    }
}

const HEDGING_WORDS: &[&str] = &[
    "might",
    "could",
    "perhaps",
    "possibly",
    "somewhat",
    "apparently",
    "seemingly",
    "arguably",
    "presumably",
    "likely",
    "maybe",
    "probably",
    "suggest",
    "suggests",
    "may",
    "seem",
    "seems",
    "sometimes",
];

const ASSERTIVE_WORDS: &[&str] = &[
    "must",
    "will",
    "always",
    "never",
    "clearly",
    "obviously",
    "definitely",
    "certainly",
    "undoubtedly",
    "absolutely",
    "every",
    "none",
    "shall",
    "undeniable",
    "unquestionably",
];

fn hedging_assertive_score(words: &[&str]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let mut hedging = 0usize;
    let mut assertive = 0usize;
    for w in words {
        let lower = w.to_lowercase();
        if HEDGING_WORDS.contains(&lower.as_str()) {
            hedging += 1;
        }
        if ASSERTIVE_WORDS.contains(&lower.as_str()) {
            assertive += 1;
        }
    }
    (hedging as f64 - assertive as f64) / words.len() as f64
}

fn avg_sentence_length_words(text: &str) -> f64 {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if sentences.is_empty() {
        return 0.0;
    }
    let total_words: usize = sentences.iter().map(|s| s.split_whitespace().count()).sum();
    total_words as f64 / sentences.len() as f64
}

fn avg_word_length_chars(words: &[&str]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let total: usize = words.iter().map(|w| w.len()).sum();
    total as f64 / words.len() as f64
}

// ── Analyzer ──────────────────────────────────────────────────

/// Stylistic analyzer detecting voice, tone, and structural changes.
pub struct StylisticAnalyzer;

impl AnalyzerPlugin for StylisticAnalyzer {
    fn id(&self) -> &str {
        "stylistic"
    }

    fn meta(&self) -> AnalyzerMeta {
        AnalyzerMeta {
            id: "stylistic".into(),
            name: "Stylistic Analyzer".into(),
            version: "1.0.0".into(),
        }
    }

    fn dependencies(&self) -> Vec<AnalyzerDependency> {
        Vec::new()
    }

    fn cost(&self) -> f32 {
        0.4
    }

    fn analyze(
        &self,
        context: &AnalysisContext,
        _prior_results: &AnalysisReport,
    ) -> Result<Box<dyn Any + Send + Sync>, AnalysisError> {
        let changes = context.change_operations();
        let mut voice_shifts = Vec::new();
        let mut tone_shifts = Vec::new();

        let src_text = &context.diff.source.normalized;
        let tgt_text = &context.diff.target.normalized;

        for (idx, op) in &changes {
            if op.kind != EditKind::Replace {
                continue;
            }

            // Extract word strings for source and target
            let src_tokens = context.source_tokens_for_op(*idx).unwrap_or(&[]);
            let tgt_tokens = context.target_tokens_for_op(*idx).unwrap_or(&[]);

            let src_words: Vec<&str> = src_tokens
                .iter()
                .filter_map(|t| {
                    let s = t.span.start() as usize;
                    let e = (t.span.end() as usize).min(src_text.len());
                    if s <= e { Some(&src_text[s..e]) } else { None }
                })
                .collect();
            let tgt_words: Vec<&str> = tgt_tokens
                .iter()
                .filter_map(|t| {
                    let s = t.span.start() as usize;
                    let e = (t.span.end() as usize).min(tgt_text.len());
                    if s <= e { Some(&tgt_text[s..e]) } else { None }
                })
                .collect();

            // Voice detection
            let src_voice = classify_voice(&src_words);
            let tgt_voice = classify_voice(&tgt_words);
            if src_voice != tgt_voice && src_voice != Voice::Unknown && tgt_voice != Voice::Unknown
            {
                let confidence = 0.7; // heuristic-based, moderate confidence
                voice_shifts.push(VoiceShift {
                    operation_index: *idx,
                    from: src_voice,
                    to: tgt_voice,
                    confidence,
                });
            }

            // Tone detection (hedging/assertive)
            let src_score = hedging_assertive_score(&src_words);
            let tgt_score = hedging_assertive_score(&tgt_words);
            let delta = (tgt_score - src_score).abs();
            if delta > 0.15 {
                tone_shifts.push(ToneShift {
                    operation_index: *idx,
                    dimension: ToneDimension::HedgingAssertive,
                    from_score: src_score,
                    to_score: tgt_score,
                    confidence: (delta * 2.0).min(0.95),
                });
            }
        }

        // Structural changes from full document
        let src_words_all: Vec<&str> = src_text.split_whitespace().collect();
        let tgt_words_all: Vec<&str> = tgt_text.split_whitespace().collect();

        let source_avg_sentence_length = avg_sentence_length_words(src_text);
        let target_avg_sentence_length = avg_sentence_length_words(tgt_text);
        let source_avg_word_length = avg_word_length_chars(&src_words_all);
        let target_avg_word_length = avg_word_length_chars(&tgt_words_all);

        let structural_changes = StructuralChanges {
            source_avg_sentence_length,
            target_avg_sentence_length,
            sentence_length_delta: target_avg_sentence_length - source_avg_sentence_length,
            source_avg_word_length,
            target_avg_word_length,
            word_length_delta: target_avg_word_length - source_avg_word_length,
        };

        Ok(Box::new(StylisticResult {
            voice_shifts,
            tone_shifts,
            structural_changes,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passive_voice_detection() {
        assert!(passive_voice_ratio(&["was", "thrown"]) > 0.0);
    }

    #[test]
    fn active_voice_detection() {
        assert!(passive_voice_ratio(&["he", "threw", "the", "ball"]).abs() < f64::EPSILON);
    }

    #[test]
    fn classify_voice_passive() {
        assert_eq!(classify_voice(&["it", "was", "broken"]), Voice::Passive);
    }

    #[test]
    fn classify_voice_active() {
        assert_eq!(classify_voice(&["they", "broke", "it"]), Voice::Active);
    }

    #[test]
    fn hedging_positive() {
        assert!(hedging_assertive_score(&["it", "might", "possibly", "work"]) > 0.0);
    }

    #[test]
    fn assertive_negative() {
        assert!(hedging_assertive_score(&["you", "must", "always", "do"]) < 0.0);
    }

    #[test]
    fn hedging_neutral() {
        assert!(hedging_assertive_score(&["the", "cat", "sat"]).abs() < f64::EPSILON);
    }

    #[test]
    fn avg_sentence_length_test() {
        let avg = avg_sentence_length_words("Hello world. How are you.");
        assert!((avg - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn avg_word_length_test() {
        let avg = avg_word_length_chars(&["hi", "there", "world"]);
        let expected = (2.0 + 5.0 + 5.0) / 3.0;
        assert!((avg - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn stylistic_analyzer_id() {
        assert_eq!(StylisticAnalyzer.id(), "stylistic");
    }

    #[test]
    fn stylistic_analyzer_no_deps() {
        assert!(StylisticAnalyzer.dependencies().is_empty());
    }
}
