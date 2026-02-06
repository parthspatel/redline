//! Error types for redline-core.
//!
//! Stub — full implementation via TDD in Task 2.

/// Top-level error type for redline-core operations.
#[derive(Debug)]
pub enum RedlineError {}

impl core::fmt::Display for RedlineError {
    fn fmt(&self, _f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {}
    }
}
