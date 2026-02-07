//! Edit operations for representing differences between token sequences.

/// The kind of edit operation in a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EditKind {
    /// Tokens are equal in source and target.
    Equal = 0,
    /// Tokens were deleted from source (not present in target).
    Delete = 1,
    /// Tokens were inserted into target (not present in source).
    Insert = 2,
    /// Tokens in source were replaced with different tokens in target.
    Replace = 3,
}

/// A single edit operation with half-open `[start, end)` token index ranges.
///
/// - `Equal`: both ranges have equal length.
/// - `Delete`: target range is empty (`target_start == target_end`).
/// - `Insert`: source range is empty (`source_start == source_end`).
/// - `Replace`: both ranges are non-empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EditOperation {
    pub kind: EditKind,
    pub source_start: u32,
    pub source_end: u32,
    pub target_start: u32,
    pub target_end: u32,
}

impl EditOperation {
    #[inline]
    pub fn equal(src_start: u32, src_end: u32, tgt_start: u32, tgt_end: u32) -> Self {
        Self {
            kind: EditKind::Equal,
            source_start: src_start,
            source_end: src_end,
            target_start: tgt_start,
            target_end: tgt_end,
        }
    }

    #[inline]
    pub fn delete(src_start: u32, src_end: u32, tgt_pos: u32) -> Self {
        Self {
            kind: EditKind::Delete,
            source_start: src_start,
            source_end: src_end,
            target_start: tgt_pos,
            target_end: tgt_pos,
        }
    }

    #[inline]
    pub fn insert(src_pos: u32, tgt_start: u32, tgt_end: u32) -> Self {
        Self {
            kind: EditKind::Insert,
            source_start: src_pos,
            source_end: src_pos,
            target_start: tgt_start,
            target_end: tgt_end,
        }
    }

    #[inline]
    pub fn replace(src_start: u32, src_end: u32, tgt_start: u32, tgt_end: u32) -> Self {
        Self {
            kind: EditKind::Replace,
            source_start: src_start,
            source_end: src_end,
            target_start: tgt_start,
            target_end: tgt_end,
        }
    }

    #[inline]
    pub fn source_len(&self) -> u32 {
        self.source_end - self.source_start
    }

    #[inline]
    pub fn target_len(&self) -> u32 {
        self.target_end - self.target_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_constructor() {
        let op = EditOperation::equal(0, 3, 0, 3);
        assert_eq!(op.kind, EditKind::Equal);
        assert_eq!(op.source_start, 0);
        assert_eq!(op.source_end, 3);
        assert_eq!(op.target_start, 0);
        assert_eq!(op.target_end, 3);
    }

    #[test]
    fn delete_constructor() {
        let op = EditOperation::delete(2, 5, 2);
        assert_eq!(op.kind, EditKind::Delete);
        assert_eq!(op.target_start, 2);
        assert_eq!(op.target_end, 2);
    }

    #[test]
    fn insert_constructor() {
        let op = EditOperation::insert(3, 3, 6);
        assert_eq!(op.kind, EditKind::Insert);
        assert_eq!(op.source_start, 3);
        assert_eq!(op.source_end, 3);
    }

    #[test]
    fn replace_constructor() {
        let op = EditOperation::replace(1, 3, 1, 4);
        assert_eq!(op.kind, EditKind::Replace);
    }

    #[test]
    fn source_len_and_target_len() {
        assert_eq!(EditOperation::equal(0, 5, 0, 5).source_len(), 5);
        assert_eq!(EditOperation::equal(0, 5, 0, 5).target_len(), 5);
        assert_eq!(EditOperation::delete(2, 7, 2).source_len(), 5);
        assert_eq!(EditOperation::delete(2, 7, 2).target_len(), 0);
        assert_eq!(EditOperation::insert(3, 3, 8).source_len(), 0);
        assert_eq!(EditOperation::insert(3, 3, 8).target_len(), 5);
    }

    #[test]
    fn edit_kind_repr_values() {
        assert_eq!(EditKind::Equal as u8, 0);
        assert_eq!(EditKind::Delete as u8, 1);
        assert_eq!(EditKind::Insert as u8, 2);
        assert_eq!(EditKind::Replace as u8, 3);
    }

    #[test]
    fn edit_operation_is_copy() {
        let op = EditOperation::equal(0, 1, 0, 1);
        let op2 = op;
        let op3 = op;
        assert_eq!(op2, op3);
    }
}
