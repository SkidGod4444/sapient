//! `Shape` — tensor shape utilities and broadcasting rules.

use crate::error::{Result, SapientError};
use serde::{Deserialize, Serialize};

/// Newtype around a dimension vector that carries shape utilities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Construct from any iterator of `usize`.
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self(dims.into_iter().collect())
    }

    /// Number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements (product of all dims).
    #[inline]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Dimension slice.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Row-major (C-contiguous) strides.
    pub fn strides(&self) -> Vec<usize> {
        let n = self.ndim();
        if n == 0 {
            return vec![];
        }
        let mut strides = vec![1usize; n];
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        strides
    }

    /// Scalar (0-dimensional) shape.
    pub fn scalar() -> Self {
        Self(vec![])
    }

    /// Whether this is a scalar.
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Reshape — ensures the total numel is unchanged.
    pub fn reshape(&self, new_dims: impl IntoIterator<Item = usize>) -> Result<Shape> {
        let new_shape = Shape::new(new_dims);
        if new_shape.numel() != self.numel() {
            return Err(SapientError::ShapeMismatch {
                expected: self.0.clone(),
                got: new_shape.0.clone(),
            });
        }
        Ok(new_shape)
    }

    /// Compute the broadcast output shape of `self` and `other` (NumPy rules).
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape> {
        let (a, b) = (&self.0, &other.0);
        let len = a.len().max(b.len());
        let mut out = vec![0usize; len];
        for i in 0..len {
            let ai = if i < len - a.len() {
                1
            } else {
                a[i - (len - a.len())]
            };
            let bi = if i < len - b.len() {
                1
            } else {
                b[i - (len - b.len())]
            };
            if ai == bi {
                out[i] = ai;
            } else if ai == 1 {
                out[i] = bi;
            } else if bi == 1 {
                out[i] = ai;
            } else {
                return Err(SapientError::BroadcastError {
                    lhs: self.0.clone(),
                    rhs: other.0.clone(),
                });
            }
        }
        Ok(Shape(out))
    }

    /// Insert a new axis of size 1 at `axis` (like `np.expand_dims`).
    pub fn expand_dims(&self, axis: usize) -> Result<Shape> {
        if axis > self.ndim() {
            return Err(SapientError::internal(format!(
                "expand_dims: axis {axis} out of range for rank {}",
                self.ndim()
            )));
        }
        let mut dims = self.0.clone();
        dims.insert(axis, 1);
        Ok(Shape(dims))
    }

    /// Remove all dimensions of size 1 (like `np.squeeze`).
    pub fn squeeze(&self) -> Shape {
        Shape(self.0.iter().copied().filter(|&d| d != 1).collect())
    }

    /// Validate that every dim is > 0.
    pub fn validate(&self) -> Result<()> {
        for (i, &d) in self.0.iter().enumerate() {
            if d == 0 {
                return Err(SapientError::InvalidGraph(format!(
                    "Shape has zero dimension at axis {i}"
                )));
            }
        }
        Ok(())
    }

    /// Contiguous byte offset for a multi-index into row-major storage.
    pub fn flat_index(&self, idx: &[usize]) -> Result<usize> {
        if idx.len() != self.ndim() {
            return Err(SapientError::RankMismatch {
                expected: self.ndim(),
                got: idx.len(),
            });
        }
        let strides = self.strides();
        let mut offset = 0;
        for (i, (&ix, &st)) in idx.iter().zip(strides.iter()).enumerate() {
            if ix >= self.0[i] {
                return Err(SapientError::internal(format!(
                    "Index {ix} out of bounds for dim {i} (size {})",
                    self.0[i]
                )));
            }
            offset += ix * st;
        }
        Ok(offset)
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Self(s.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel() {
        assert_eq!(Shape::new([2, 3, 4]).numel(), 24);
        assert_eq!(Shape::scalar().numel(), 1);
    }

    #[test]
    fn strides_row_major() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn broadcast() {
        let a = Shape::new([1, 3]);
        let b = Shape::new([2, 3]);
        assert_eq!(a.broadcast_with(&b).unwrap(), Shape::new([2, 3]));
    }

    #[test]
    fn broadcast_fail() {
        let a = Shape::new([2, 3]);
        let b = Shape::new([2, 4]);
        assert!(a.broadcast_with(&b).is_err());
    }

    #[test]
    fn reshape() {
        let s = Shape::new([2, 3]);
        let r = s.reshape([6]).unwrap();
        assert_eq!(r, Shape::new([6]));
    }

    #[test]
    fn flat_index() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.flat_index(&[1, 2, 3]).unwrap(), 12 + 8 + 3);
    }
}
