// Sources:
// - http://wscg.zcu.cz/wscg2012/short/A29-full.pdf
// - https://github.com/bobbens/libdq/blob/master/dq.h#L293

use crate::{
    matrix::Matrix,
    num::{ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub, One, Sqrt, Zero},
    rotation::quaternion::Quaternion,
    vector::Vector,
};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DualQuaternion<T> {
    real: Quaternion<T>,
    dual: Quaternion<T>,
}

impl<T: Zero + One> DualQuaternion<T> {
    #[must_use]
    #[inline]
    pub const fn identity() -> Self {
        Self {
            real: Quaternion::new(T::ZERO, T::ZERO, T::ZERO, T::ONE),
            dual: Quaternion::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO),
        }
    }
}

impl<T: ClosedMul + Copy + ClosedAdd + ClosedDiv + Zero + Sqrt> DualQuaternion<T> {
    #[must_use]
    #[inline]
    pub fn new(real: Quaternion<T>, dual: Quaternion<T>) -> Self {
        Self {
            real: real.normalized(),
            dual,
        }
    }

    #[must_use]
    #[inline]
    pub fn from_rotation(rotation: Quaternion<T>) -> Self {
        Self::new(rotation, Quaternion::splat(T::ZERO))
    }
}

impl<T: ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + ClosedMul + Sqrt + One + Zero>
    DualQuaternion<T>
{
    #[must_use]
    #[inline]
    pub fn from_transform(offset: Vector<T, 3>, rotation: Quaternion<T>) -> Self {
        let real = rotation.normalized();

        let dual = {
            let q = Quaternion::from_components(offset, T::ZERO) * real;
            q / (T::ONE + T::ONE)
        };

        Self { real, dual }
    }
}

impl<T: Zero + One> Default for DualQuaternion<T> {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl<T> DualQuaternion<T> {
    #[must_use]
    #[inline]
    pub fn dot<U>(self, rhs: DualQuaternion<U>) -> T::Output
    where
        T: Mul<U>,
        T::Output: Zero + ClosedAdd,
    {
        Quaternion::dot(self.real, rhs.real)
    }

    #[must_use]
    #[inline]
    pub const fn rotation(&self) -> &Quaternion<T> {
        &self.real
    }
}

impl<T: ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + ClosedMul + Sqrt + Zero + One>
    DualQuaternion<T>
{
    #[must_use]
    #[inline]
    pub fn to_translation_vector(self) -> Vector<T, 3> {
        let t = (self.dual * (T::ONE + T::ONE)) * Quaternion::conjugated(self.real);
        t.v
    }
}

impl<T: ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + ClosedMul + Sqrt + Zero + One>
    DualQuaternion<T>
{
    #[must_use]
    #[inline]
    pub fn to_matrix(self) -> Matrix<T, 4, 4> {
        // Extract the translation.
        let trans = {
            let [x, y, z] = Self::to_translation_vector(self).to_array();
            [x, y, z, T::ONE]
        };

        // Extract the rotation.
        let rotation = self.real;

        let mut matrix = Matrix::rotation_3d(rotation);
        matrix.set_col(3, trans);
        matrix
    }
}

impl<T: Copy + ClosedAdd + DivAssign + ClosedMul + Zero> DualQuaternion<T> {
    #[inline]
    pub fn normalized(self) -> Self {
        let mag = DualQuaternion::dot(self, self);
        let mut norm = self;
        norm /= mag;
        norm
    }

    #[inline]
    pub fn normalize(&mut self) {
        *self = DualQuaternion::normalized(*self);
    }
}

impl<T: ClosedNeg> DualQuaternion<T> {
    #[must_use]
    #[inline]
    pub fn conjugated(self) -> Self {
        Self {
            real: Quaternion::conjugated(self.real),
            dual: Quaternion::conjugated(self.dual),
        }
    }

    #[inline]
    pub fn conjugate(&mut self) {
        self.real.conjugate();
        self.dual.conjugate();
    }
}

impl<T: ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + ClosedMul + Sqrt + Zero> Mul
    for DualQuaternion<T>
{
    type Output = DualQuaternion<T>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real * rhs.real,
            dual: rhs.dual * self.real + rhs.real * self.dual,
        }
    }
}

impl<T: ClosedAdd> Add for DualQuaternion<T> {
    type Output = DualQuaternion<T>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        DualQuaternion {
            real: self.real.add(rhs.real),
            dual: self.dual.add(rhs.dual),
        }
    }
}

impl<T: AddAssign> AddAssign for DualQuaternion<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.real.add_assign(rhs.real);
        self.dual.add_assign(rhs.dual);
    }
}

impl<T: ClosedSub> Sub for DualQuaternion<T> {
    type Output = DualQuaternion<T>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        DualQuaternion {
            real: self.real.sub(rhs.real),
            dual: self.dual.sub(rhs.dual),
        }
    }
}

impl<T: Mul + Copy> Mul<T> for DualQuaternion<T> {
    type Output = DualQuaternion<T::Output>;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        DualQuaternion {
            real: self.real * rhs,
            dual: self.dual * rhs,
        }
    }
}

impl<T: MulAssign + Copy> MulAssign<T> for DualQuaternion<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.real.mul_assign(rhs);
        self.dual.mul_assign(rhs);
    }
}

impl<T: Div + Copy> Div<T> for DualQuaternion<T> {
    type Output = DualQuaternion<T::Output>;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        DualQuaternion {
            real: self.real / rhs,
            dual: self.dual / rhs,
        }
    }
}

impl<T: DivAssign + Copy> DivAssign<T> for DualQuaternion<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.real.div_assign(rhs);
        self.dual.div_assign(rhs);
    }
}

impl<T: SubAssign> SubAssign for DualQuaternion<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.real.sub_assign(rhs.real);
        self.dual.sub_assign(rhs.dual);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quaternion::Quaternion, rotation::angle::Angle};
    use approx::assert_relative_eq;

    #[test]
    fn test_transforms() {
        let rotation = Quaternion::from_angle_axis(
            Angle::Degrees(-32.0),
            Vector::new([1.0, 4.0, 2.0]).normalized(),
        );

        let offset = Vector::new([23.0, 45.0, 200.0]);

        let matrix = {
            let translation = Matrix::translation_3d(offset);
            let rotation = Matrix::rotation_3d(rotation);
            translation * rotation
        };

        let dual_quat = DualQuaternion::from_transform(offset, rotation);
        assert_relative_eq!(dual_quat.to_matrix(), matrix, epsilon = 1e-14);
    }
}
