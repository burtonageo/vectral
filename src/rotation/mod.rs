// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "nightly")]
use crate::{
    matrix::Matrix,
    num::{ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, Zero},
    point::Point,
    vector::Vector,
};

pub mod angle;
pub mod quaternion;

#[cfg(feature = "nightly")]
pub trait Rotation<const DIM: usize> {
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

#[cfg(feature = "nightly")]
#[must_use]
#[inline]
pub fn rotate_point_around<T, R, const N: usize>(
    point_to_rotate: Point<T, N>,
    center_of_rotation: Point<T, N>,
    rotation: R,
) -> Point<T, N>
where
    R: Rotation<N, Scalar = T>,
    T: Copy + ClosedDiv + ClosedSub + ClosedMul + ClosedAdd + Zero,
{
    let dir = point_to_rotate.vector_to(center_of_rotation);
    let rotated_dir = rotation.transform_vector(dir);
    point_to_rotate + dir - rotated_dir
}
