use ndarray::prelude::*;
use ndarray::OwnedRepr;

// Re-export the crate Error.
pub use crate::error::Error;

// Alias Result to be the crate Result.
pub type Result<T> = core::result::Result<T, Error>;

// Generic Wrapper tuple struct for newtype pattern,
// mostly for external type to type From/TryFrom conversions
pub struct W<T>(pub T);

impl From<Vec<f64>> for W<Array1<f64>> {
    fn from(v: Vec<f64>) -> Self {
        Self(Array1::from(v))
    }
}