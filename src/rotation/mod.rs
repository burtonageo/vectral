// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "nightly")]
use crate::matrix::Matrix;
use crate::vector::Vector;

pub mod angle;
pub mod quaternion {
    pub use crate::quaternion::*;
}

pub trait Rotation<const DIM: usize> {
    type Scalar;

    #[must_use]
    fn identity() -> Self;

    #[must_use]
    fn inverse(self) -> Self;

    #[must_use]
    fn transform_vector(&self, vector: Vector<Self::Scalar, DIM>) -> Vector<Self::Scalar, DIM>;

    #[must_use]
    fn slerp(self, target: Self, time: Self::Scalar) -> Self;
}

#[cfg(feature = "nightly")]
pub trait HomogenousRotation<const DIM: usize>: Rotation<DIM> {
    #[must_use]
    fn from_homogeneous(matrix: Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>) -> Self;

    #[must_use]
    fn into_homogeneous(self) -> Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>;

    #[must_use]
    fn get_homogeneous(&self) -> Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>;
}
