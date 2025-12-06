use crate::{matrix::Matrix, point::Point, vector::Vector, rotation::Rotation};

pub mod matrix_transform;

pub trait Transform<const DIM: usize>
where
    Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    type Scalar;
    type Rotation: Rotation<DIM, Scalar = Self::Scalar>;

    #[must_use]
    fn identity() -> Self;

    #[must_use]
    fn from_components(
        translation: Vector<Self::Scalar, DIM>,
        scale: Vector<Self::Scalar, DIM>,
        rotation: Self::Rotation,
    ) -> Self;

    #[must_use]
    fn into_components(
        self,
    ) -> (
        Vector<Self::Scalar, DIM>,
        Vector<Self::Scalar, DIM>,
        Self::Rotation,
    );

    #[must_use]
    fn get_homogeneous(&self) -> Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>;

    #[must_use]
    fn get_inverse_homogeneous(&self) -> Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>;

    #[must_use]
    fn transform_point(&self, point: Point<Self::Scalar, DIM>) -> Point<Self::Scalar, DIM>;

    #[must_use]
    fn transform_vector(&self, vector: Vector<Self::Scalar, DIM>) -> Vector<Self::Scalar, DIM>;
}

pub trait Translate<const DIM: usize>
where
    Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    type Scalar;
    #[must_use]
    fn translated<T: Transform<DIM, Scalar = Self::Scalar>>(&self, transform: &T) -> Self;

    fn translate_by<T: Transform<DIM, Scalar = Self::Scalar>>(&mut self, transform: &T);
}
