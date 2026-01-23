use crate::{
    matrix::{Matrix, Matrix4, TransformHomogeneous},
    point::Point,
    rotation::{angle::Angle, quaternion::Quaternion},
    transform::{Transform, Translate},
    vector::{Vector, Vector3},
};
use crate::utils::num::{
    checked::{CheckedAddAssign, CheckedDiv}, Abs, Bounded, ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub, One, Sqrt, Trig, Zero
};
use core::ops::{AddAssign, DivAssign, Neg, SubAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatrixTransform<T = f32, const DIM: usize = 3>
where
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    matrix: Matrix<T, { DIM + 1 }, { DIM + 1 }>,
    inverse: Matrix<T, { DIM + 1 }, { DIM + 1 }>,
}

impl<T, const DIM: usize> MatrixTransform<T, DIM>
where
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    #[must_use]
    #[inline]
    pub const fn new_unchecked(
        matrix: Matrix<T, { DIM + 1 }, { DIM + 1 }>,
        inverse: Matrix<T, { DIM + 1 }, { DIM + 1 }>,
    ) -> Self {
        Self { matrix, inverse }
    }

    #[must_use]
    #[inline]
    pub const fn matrix(&self) -> &Matrix<T, { DIM + 1 }, { DIM + 1 }> {
        &self.matrix
    }

    #[must_use]
    #[inline]
    pub const fn inverse_matrix(&self) -> &Matrix<T, { DIM + 1 }, { DIM + 1 }> {
        &self.inverse
    }

    #[must_use]
    #[inline]
    pub fn inverse_transform(self) -> Self {
        let Self { matrix, inverse } = self;
        Self {
            matrix: inverse,
            inverse: matrix,
        }
    }
}

impl<T> MatrixTransform<T, 3>
where
    T: AddAssign + SubAssign + Zero + Copy + ClosedAdd + ClosedMul + ClosedSub + PartialOrd,
{
    #[must_use]
    #[inline]
    pub fn swaps_handedness(&self) -> bool {
        self.matrix.cofactor(3, 3).determinant() < T::ZERO
    }
}

impl<T, const DIM: usize> MatrixTransform<T, DIM>
where
    T: AddAssign
        + SubAssign
        + Zero
        + Copy
        + ClosedAdd
        + ClosedMul
        + ClosedSub
        + PartialEq
        + ClosedNeg
        + ClosedDiv
        + One,
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    #[must_use]
    #[inline]
    pub fn new(matrix: Matrix<T, { DIM + 1 }, { DIM + 1 }>) -> Option<Self> {
        Matrix::inverse_checked(matrix).map(|inverse| Self { inverse, matrix })
    }
}

impl<T: Copy + ClosedNeg + One + Zero, const DIM: usize> MatrixTransform<T, DIM>
where
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    #[must_use]
    #[inline]
    pub fn translation(delta: Vector<T, DIM>) -> Self {
        let mut matrix = Matrix::identity();
        let mut inverse = Matrix::identity();

        let delta = delta.expand(T::ONE);

        matrix.set_col(DIM - 1, delta.to_array());
        inverse.set_col(DIM - 1, delta.neg().to_array());

        Self { matrix, inverse }
    }
}

impl<T: Copy + ClosedDiv + One + Zero, const DIM: usize> MatrixTransform<T, DIM>
where
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    #[must_use]
    #[inline]
    pub fn scaling(scale: Vector<T, DIM>) -> Self {
        let mut matrix = Matrix::identity();
        let mut inverse = Matrix::identity();

        let scale: Vector<T, _> = scale.expand(T::ONE);

        matrix.set_rightwards_diagonal(scale.to_array());
        inverse.set_rightwards_diagonal(scale.map(|elem| T::ONE / elem).to_array());

        Self { matrix, inverse }
    }
}

impl<T: Zero + One + Trig + Copy + ClosedNeg> MatrixTransform<T> {
    #[must_use]
    #[inline]
    pub fn x_axis_rotation(angle: Angle<T>) -> Self {
        let rotation = Matrix4::x_axis_rotation(angle);
        let inverse = Matrix4::transpose(rotation);
        Self::new_unchecked(rotation, inverse)
    }

    #[must_use]
    #[inline]
    pub fn y_axis_rotation(angle: Angle<T>) -> Self {
        let rotation = Matrix4::y_axis_rotation(angle);
        let inverse = Matrix4::transpose(rotation);
        Self::new_unchecked(rotation, inverse)
    }

    #[must_use]
    #[inline]
    pub fn z_axis_rotation(angle: Angle<T>) -> Self {
        let rotation = Matrix4::z_axis_rotation(angle);
        let inverse = Matrix4::transpose(rotation);
        Self::new_unchecked(rotation, inverse)
    }
}

impl<T: Zero + ClosedAdd + ClosedMul + ClosedDiv + ClosedSub + Sqrt + Trig + One>
    MatrixTransform<T>
{
    #[must_use]
    #[inline]
    pub fn axis_rotation(angle: Angle<T>, axis: Vector3<T>) -> Self {
        let rotation = Matrix4::axis_rotation_3d(angle, axis);
        let inverse = Matrix4::transpose(rotation);
        Self::new_unchecked(rotation, inverse)
    }
}

impl<T: Zero + One, const DIM: usize> Default for MatrixTransform<T, DIM>
where
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    #[inline]
    fn default() -> Self {
        Self::new_unchecked(Matrix::identity(), Matrix::identity())
    }
}

impl<T> Transform<3> for MatrixTransform<T, 3>
where
    T: Abs
        + AddAssign
        + Bounded
        + CheckedAddAssign
        + CheckedDiv
        + ClosedNeg
        + ClosedAdd
        + ClosedDiv
        + ClosedMul
        + ClosedSub
        + Copy
        + DivAssign
        + One
        + PartialEq
        + PartialOrd
        + Sqrt
        + SubAssign
        + Trig
        + Zero,
{
    type Scalar = T;

    type Rotation = Quaternion<T>;

    #[inline]
    fn identity() -> Self {
        MatrixTransform::new_unchecked(Matrix::identity(), Matrix::identity())
    }

    #[inline]
    fn from_components(
        translation: Vector<Self::Scalar, 3>,
        scale: Vector<Self::Scalar, 3>,
        rotation: Self::Rotation,
    ) -> Self {
        let translation = Matrix::translation(translation);
        let scale = Matrix::scaling(scale);
        let rotation = Matrix::rotation_3d(rotation);

        MatrixTransform::new(translation * scale * rotation).unwrap()
    }

    #[inline]
    fn into_components(
        self,
    ) -> (
        Vector<Self::Scalar, 3>,
        Vector<Self::Scalar, 3>,
        Self::Rotation,
    ) {
        let (translation, scale, rotation, ..) =
            Matrix::decompose_homogeneous_transform_3d::<Self::Rotation>(self.matrix);
        (translation, scale, rotation)
    }

    #[inline]
    fn get_homogeneous(&self) -> Matrix<Self::Scalar, { 3 + 1 }, { 3 + 1 }> {
        self.matrix
    }

    #[inline]
    fn get_inverse_homogeneous(&self) -> Matrix<Self::Scalar, { 3 + 1 }, { 3 + 1 }> {
        self.inverse
    }

    #[inline]
    fn transform_point(&self, point: Point<Self::Scalar, 3>) -> Point<Self::Scalar, 3> {
        TransformHomogeneous::transform_homogeneous(point, self.matrix)
    }

    #[inline]
    fn transform_vector(&self, vector: Vector<Self::Scalar, 3>) -> Vector<Self::Scalar, 3> {
        TransformHomogeneous::transform_homogeneous(vector, self.matrix)
    }
}

impl<T: Copy + ClosedAdd + ClosedMul + Zero, const DIM: usize> Translate<DIM>
    for MatrixTransform<T, DIM>
where
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    type Scalar = T;
    #[inline]
    fn translated<Trans: Transform<DIM, Scalar = Self::Scalar>>(&self, transform: &Trans) -> Self {
        Self {
            matrix: transform.get_homogeneous() * self.matrix,
            inverse: transform.get_inverse_homogeneous() * self.inverse,
        }
    }

    #[inline]
    fn translate_by<Trans: Transform<DIM, Scalar = Self::Scalar>>(&mut self, transform: &Trans) {
        self.matrix = transform.get_homogeneous() * self.matrix;
        self.inverse = transform.get_inverse_homogeneous() * self.inverse;
    }
}

impl<T, const DIM: usize> TransformHomogeneous<DIM> for MatrixTransform<T, DIM>
where
    T: AddAssign
        + SubAssign
        + Zero
        + Copy
        + ClosedAdd
        + ClosedMul
        + ClosedSub
        + PartialEq
        + ClosedNeg
        + ClosedDiv
        + One,
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    type Scalar = T;

    #[inline]
    fn transform_homogeneous(self, matrix: Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>) -> Self {
        let inverse = matrix.inverse();
        Self {
            matrix: matrix * self.matrix,
            inverse: inverse * self.inverse,
        }
    }
}
