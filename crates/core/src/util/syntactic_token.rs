/// A syntactic token with POS and dependency information
#[derive(Debug, Clone)]
pub struct SyntacticToken {
    pub text: String,
    pub lemma: String,
    pub pos: String,    // Part-of-speech tag (coarse-grained)
    pub tag: String,    // Part-of-speech tag (fine-grained)
    pub dep: String,    // Dependency relation
    pub head: usize,    // Index of head token
    pub is_stop: bool,  // Is this a stop word?
    pub is_punct: bool, // Is this punctuation?
}