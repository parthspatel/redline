/// Utility functions for token statistics and distribution analysis
use crate::util::SyntacticToken;
use std::collections::HashMap;

/// Get the distribution of token tags (e.g., POS or dependency tags)
///
/// # Arguments
/// * `tokens` - Slice of syntactic tokens
/// * `tag_extractor` - Function to extract the tag from each token
///
/// # Returns
/// HashMap mapping tags to their counts
pub fn get_token_tag_distribution<F>(
    tokens: &[SyntacticToken],
    tag_extractor: F,
) -> HashMap<String, usize>
where
    F: Fn(&SyntacticToken) -> &str,
{
    let mut distribution = HashMap::new();
    for token in tokens {
        if !token.is_punct {
            *distribution
                .entry(tag_extractor(token).to_string())
                .or_insert(0) += 1;
        }
    }
    distribution
}

/// Calculate the divergence between two token tag distributions
/// Uses a symmetric divergence measure similar to Jensen-Shannon divergence
///
/// # Arguments
/// * `orig` - Original tokens
/// * `modified` - Modified tokens
/// * `tag_extractor` - Function to extract the tag from each token
///
/// # Returns
/// Divergence value in range [0.0, 1.0] where 0.0 means identical distributions
pub fn calculate_tag_distribution_divergence<F>(
    orig: &[SyntacticToken],
    modified: &[SyntacticToken],
    tag_extractor: F,
) -> f64
where
    F: Fn(&SyntacticToken) -> &str,
{
    if orig.is_empty() && modified.is_empty() {
        return 0.0;
    }
    if orig.is_empty() || modified.is_empty() {
        return 1.0;
    }

    // Count tags
    let mut orig_counts: HashMap<String, usize> = HashMap::new();
    let mut mod_counts: HashMap<String, usize> = HashMap::new();

    for token in orig {
        *orig_counts
            .entry(tag_extractor(token).to_string())
            .or_insert(0) += 1;
    }

    for token in modified {
        *mod_counts
            .entry(tag_extractor(token).to_string())
            .or_insert(0) += 1;
    }

    // Normalize to probabilities
    let orig_total = orig.len() as f64;
    let mod_total = modified.len() as f64;

    // Calculate symmetric divergence
    let all_tags: std::collections::HashSet<String> = orig_counts
        .keys()
        .chain(mod_counts.keys())
        .cloned()
        .collect();

    let mut divergence = 0.0;

    for tag in all_tags {
        let p_orig = *orig_counts.get(&tag).unwrap_or(&0) as f64 / orig_total;
        let p_mod = *mod_counts.get(&tag).unwrap_or(&0) as f64 / mod_total;

        // Avoid log(0) by using absolute difference
        if p_orig > 0.0 && p_mod > 0.0 {
            divergence += (p_orig - p_mod).abs();
        } else {
            divergence += p_orig.max(p_mod);
        }
    }

    // Normalize to [0, 1]
    divergence / 2.0
}

/// Calculate the ratio of unchanged tokens between original and modified
///
/// # Arguments
/// * `orig` - Original tokens
/// * `modified` - Modified tokens
/// * `comparator` - Function to compare two tokens for equality
///
/// # Returns
/// Ratio in range [0.0, 1.0] where 1.0 means all tokens are unchanged
pub fn calculate_unchanged_ratio<F>(
    orig: &[SyntacticToken],
    modified: &[SyntacticToken],
    comparator: F,
) -> f64
where
    F: Fn(&SyntacticToken, &SyntacticToken) -> bool,
{
    let min_len = orig.len().min(modified.len());
    if min_len == 0 {
        return 1.0;
    }

    let unchanged = (0..min_len)
        .filter(|&i| comparator(&orig[i], &modified[i]))
        .count();

    unchanged as f64 / min_len as f64
}

/// Categorize change magnitude into human-readable descriptions
///
/// # Arguments
/// * `count` - Number of changes
/// * `change_type` - Type of change (e.g., "POS", "dependency")
///
/// # Returns
/// Human-readable description of the change magnitude
pub fn categorize_change_magnitude(count: usize, change_type: &str) -> String {
    match count {
        0 => format!("No {} changes detected", change_type),
        1..=2 => format!("Minimal {} changes: {} changed", change_type, count),
        3..=5 => format!("Moderate {} changes: {} changed", change_type, count),
        _ => format!("Significant {} changes: {} changed", change_type, count),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categorize_change_magnitude() {
        assert_eq!(
            categorize_change_magnitude(0, "POS"),
            "No POS changes detected"
        );
        assert_eq!(
            categorize_change_magnitude(2, "POS"),
            "Minimal POS changes: 2 changed"
        );
        assert_eq!(
            categorize_change_magnitude(5, "POS"),
            "Moderate POS changes: 5 changed"
        );
        assert_eq!(
            categorize_change_magnitude(10, "POS"),
            "Significant POS changes: 10 changed"
        );
    }

    #[test]
    fn test_calculate_unchanged_ratio() {
        let tokens1 = vec![
            SyntacticToken {
                text: "a".to_string(),
                pos: "DET".to_string(),
                tag: "DT".to_string(),
                dep: "det".to_string(),
                lemma: "a".to_string(),
                head: 1,
                is_stop: true,
                is_punct: false,
            },
            SyntacticToken {
                text: "test".to_string(),
                pos: "NOUN".to_string(),
                tag: "NN".to_string(),
                dep: "nsubj".to_string(),
                lemma: "test".to_string(),
                head: 0,
                is_stop: false,
                is_punct: false,
            },
        ];

        let tokens2 = vec![
            SyntacticToken {
                text: "a".to_string(),
                pos: "DET".to_string(),
                tag: "DT".to_string(),
                dep: "det".to_string(),
                lemma: "a".to_string(),
                head: 1,
                is_stop: true,
                is_punct: false,
            },
            SyntacticToken {
                text: "test".to_string(),
                pos: "VERB".to_string(), // Changed
                tag: "VB".to_string(),
                dep: "nsubj".to_string(),
                lemma: "test".to_string(),
                head: 0,
                is_stop: false,
                is_punct: false,
            },
        ];

        let ratio = calculate_unchanged_ratio(&tokens1, &tokens2, |a, b| a.pos == b.pos);
        assert_eq!(ratio, 0.5);
    }
}
