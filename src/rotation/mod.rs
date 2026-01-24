#[cfg(feature = "nightly")]
use crate::{matrix::Matrix, point::Point, vector::Vector};

pub mod angle;
pub mod quaternion;

#[cfg(feature = "nightly")]
pub trait Rotation<const DIM: usize>
where
    Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    type Scalar;

    #[must_use]
    fn identity() -> Self;

    #[must_use]
    fn inverse(self) -> Self;

    #[must_use]
    fn transform_point(&self, point: Point<Self::Scalar, DIM>) -> Point<Self::Scalar, DIM>;

    #[must_use]
    fn transform_vector(&self, vector: Vector<Self::Scalar, DIM>) -> Vector<Self::Scalar, DIM>;

    #[must_use]
    fn slerp(self, target: Self, time: Self::Scalar) -> Self;

    #[must_use]
    fn from_homogeneous(matrix: Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>) -> Self;

    #[must_use]
    fn into_homogeneous(self) -> Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>;

    #[must_use]
    fn get_homogeneous(&self) -> Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>;
}
