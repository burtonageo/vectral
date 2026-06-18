// SPDX-License-Identifier: MIT OR Apache-2.0

// Sources:
// - http://wscg.zcu.cz/wscg2012/short/A29-full.pdf
// - https://github.com/bobbens/libdq/blob/master/dq.h#L293

use crate::{
    matrix::Matrix,
    num::{
        ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub, One, Sqrt, Zero,
        checked::{CheckedDiv, CheckedMul},
    },
    point::Point,
    rotation::quaternion::Quaternion,
    vector::Vector,
};
#[cfg(feature = "serde")]
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
#[cfg(feature = "serde")]
use serde_core::de;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DualQuaternion<T> {
    pub real: Quaternion<T>,
    pub dual: Quaternion<T>,
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

impl<T: Copy + One + Zero + ClosedAdd + ClosedDiv> DualQuaternion<T> {
    #[must_use]
    #[inline]
    pub fn from_position(point: Point<T, 3>) -> Self {
        let [x, y, z] = (point / (T::ONE + T::ONE)).to_array();
        let dual = Quaternion::new(x, y, z, T::ZERO);

        Self {
            real: Quaternion::identity(),
            dual,
        }
    }
}

impl<T> DualQuaternion<T>
where
    T: Copy + ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + ClosedMul + Sqrt + One + Zero,
{
    #[must_use]
    #[inline]
    pub fn from_position_rotation(position: Vector<T, 3>, rotation: Quaternion<T>) -> Self {
        let real = rotation.normalized();

        let dual = {
            let position = position / (T::ONE + T::ONE);
            Quaternion::from_components(position, T::ZERO) * real
        };

        Self { real, dual }
    }
}

impl<T> DualQuaternion<T>
where
    T: ClosedAdd
        + ClosedNeg
        + Copy
        + ClosedDiv
        + ClosedMul
        + ClosedSub
        + One
        + PartialOrd
        + Zero
        + Sqrt,
{
    #[inline]
    pub fn from_matrix(matrix: Matrix<T, 4, 4>) -> Self {
        let translation = {
            let [x, y, z, ..] = matrix.col(3);
            Vector::new([x, y, z])
        };

        let rotation = Quaternion::from_matrix(matrix);
        Self::from_position_rotation(translation, rotation)
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

impl<T> DualQuaternion<T>
where
    T: Copy + ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + ClosedMul + Sqrt + Zero + One,
{
    #[must_use]
    #[inline]
    pub fn to_translation_vector(self) -> Vector<T, 3> {
        let t = (self.dual * (T::ONE + T::ONE)) * Quaternion::conjugated(self.real);
        t.v
    }

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
    #[must_use]
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

impl<T: Copy + ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + ClosedMul + Sqrt + Zero> Mul
    for DualQuaternion<T>
{
    type Output = DualQuaternion<T>;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real * rhs.real,
            dual: (rhs.dual * self.real) + (rhs.real * self.dual),
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

impl<T: Copy + CheckedMul> CheckedMul<T> for DualQuaternion<T> {
    #[inline]
    fn checked_mul(self, rhs: T) -> Option<Self::Output> {
        Some(DualQuaternion {
            real: self.real.checked_mul(rhs)?,
            dual: self.dual.checked_mul(rhs)?,
        })
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

impl<T: Copy + CheckedDiv> CheckedDiv<T> for DualQuaternion<T> {
    #[inline]
    fn checked_div(self, rhs: T) -> Option<Self::Output> {
        Some(DualQuaternion {
            real: self.real.checked_div(rhs)?,
            dual: self.dual.checked_div(rhs)?,
        })
    }
}

impl<T: SubAssign> SubAssign for DualQuaternion<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.real.sub_assign(rhs.real);
        self.dual.sub_assign(rhs.dual);
    }
}

impl<T> From<(Quaternion<T>, Quaternion<T>)> for DualQuaternion<T>
where
    T: ClosedAdd + ClosedMul + ClosedDiv + Copy + Sqrt + Zero,
{
    #[inline]
    fn from((real, dual): (Quaternion<T>, Quaternion<T>)) -> Self {
        Self::new(real, dual)
    }
}

impl<T> From<[Quaternion<T>; 2]> for DualQuaternion<T>
where
    T: ClosedAdd + ClosedMul + ClosedDiv + Copy + Sqrt + Zero,
{
    #[inline]
    fn from([real, dual]: [Quaternion<T>; 2]) -> Self {
        Self::new(real, dual)
    }
}

impl<T> From<Matrix<T, 4, 4>> for DualQuaternion<T>
where
    T: ClosedAdd
        + ClosedNeg
        + Copy
        + ClosedDiv
        + ClosedMul
        + ClosedSub
        + One
        + PartialOrd
        + Zero
        + Sqrt,
{
    #[inline]
    fn from(value: Matrix<T, 4, 4>) -> Self {
        DualQuaternion::from_matrix(value)
    }
}

#[cfg(feature = "serde")]
impl<T: serde_core::Serialize> serde_core::Serialize for DualQuaternion<T> {
    #[inline]
    fn serialize<S: serde_core::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            use serde_core::ser::SerializeStruct;

            let mut s = serializer.serialize_struct("DualQuaternion", 2)?;
            s.serialize_field("real", &self.real)?;
            s.serialize_field("dual", &self.dual)?;
            s.end()
        } else {
            use serde_core::ser::SerializeTuple;

            let mut s = serializer.serialize_tuple(2)?;
            s.serialize_element(&self.real)?;
            s.serialize_element(&self.dual)?;
            s.end()
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, T: de::Deserialize<'de>> de::Deserialize<'de> for DualQuaternion<T> {
    #[inline]
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use core::fmt;

        enum Field {
            Real,
            Dual,
        }

        impl<'de> de::Deserialize<'de> for Field {
            #[inline]
            fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct FieldVisitor;

                impl de::Visitor<'_> for FieldVisitor {
                    type Value = Field;

                    #[inline]
                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`real` or `dual`")
                    }

                    #[inline]
                    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                        match v {
                            "real" => Ok(Field::Real),
                            "dual" => Ok(Field::Dual),
                            _ => return Err(de::Error::unknown_field(v, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct Visitor<T>(PhantomData<DualQuaternion<T>>);

        impl<'de, T: de::Deserialize<'de>> de::Visitor<'de> for Visitor<T> {
            type Value = DualQuaternion<T>;
            #[inline]
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct DualQuaternion")
            }

            #[inline]
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                collect_with!(seq, &self, [real, dual], Ok(DualQuaternion { real, dual }))
            }

            #[inline]
            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut fields = [None, None];

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Real => {
                            if fields[0].is_some() {
                                return Err(de::Error::duplicate_field("real"));
                            }
                            fields[0] = Some(map.next_value::<Quaternion<T>>()?);
                        }
                        Field::Dual => {
                            if fields[1].is_some() {
                                return Err(de::Error::duplicate_field("dual"));
                            }
                            fields[1] = Some(map.next_value::<Quaternion<T>>()?);
                        }
                    }
                }

                let [real, dual] = fields;
                let real = real.ok_or_else(|| de::Error::missing_field("real"))?;
                let dual = dual.ok_or_else(|| de::Error::missing_field("dual"))?;

                Ok(DualQuaternion { real, dual })
            }
        }

        const FIELDS: &'static [&'static str] = &["real", "dual"];

        if deserializer.is_human_readable() {
            deserializer.deserialize_struct("DualQuaternion", FIELDS, Visitor::<T>(PhantomData))
        } else {
            deserializer.deserialize_seq(Visitor::<T>(PhantomData))
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<(mint::Quaternion<T>, mint::Quaternion<T>)> for DualQuaternion<T>
where
    T: ClosedAdd + ClosedMul + ClosedDiv + Copy + Sqrt + Zero,
{
    #[inline]
    fn from((real, dual): (mint::Quaternion<T>, mint::Quaternion<T>)) -> Self {
        Self::new(From::from(real), From::from(dual))
    }
}

#[cfg(feature = "mint")]
impl<T> From<[mint::Quaternion<T>; 2]> for DualQuaternion<T>
where
    T: ClosedAdd + ClosedMul + ClosedDiv + Copy + Sqrt + Zero,
{
    #[inline]
    fn from([real, dual]: [mint::Quaternion<T>; 2]) -> Self {
        Self::new(From::from(real), From::from(dual))
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for DualQuaternion<T> {}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for DualQuaternion<T> {}

#[cfg(feature = "approx")]
impl<T: approx::AbsDiffEq> approx::AbsDiffEq for DualQuaternion<T>
where
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real.abs_diff_eq(&other.real, epsilon.clone())
            && self.dual.abs_diff_eq(&other.dual, epsilon)
    }
}

#[cfg(feature = "approx")]
impl<T: approx::RelativeEq> approx::RelativeEq for DualQuaternion<T>
where
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.real
            .relative_eq(&other.real, epsilon.clone(), max_relative.clone())
            && self.dual.relative_eq(&other.dual, epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl<T: approx::UlpsEq> approx::UlpsEq for DualQuaternion<T>
where
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.real.ulps_eq(&other.real, epsilon.clone(), max_ulps)
            && self.dual.ulps_eq(&other.dual, epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quaternion::Quaternion, rotation::angle::Angle};
    use approx::assert_relative_eq;

    #[test]
    fn test_transforms() {
        let rotation_1 = Quaternion::from_angle_axis(
            Angle::Degrees(-32.0),
            Vector::new([1.0, 4.0, 2.0]).normalized(),
        );

        let rotation_2 = Quaternion::from_angle_axis(
            Angle::Degrees(-142.7),
            Vector::new([3.0, -1.0, 2.7]).normalized(),
        );

        let offset = Vector::new([23.0, 45.0, 200.0]);

        let matrix = {
            Matrix::translation_3d(offset)
                * Matrix::rotation_3d(rotation_1)
                * Matrix::rotation_3d(rotation_2)
        };

        let mut dual_quat: DualQuaternion<f64> = DualQuaternion::from_rotation(rotation_1)
            * DualQuaternion::from_rotation(rotation_2)
            * DualQuaternion::from_position(offset.into());
        assert_relative_eq!(dual_quat.to_matrix(), matrix, epsilon = 1e-13);

        dual_quat.normalize();
        assert_relative_eq!(dual_quat.to_matrix(), matrix, epsilon = 1e-13);

        let pure_translation = DualQuaternion::from_position(offset.into());
        let trans_matrix = Matrix::translation_3d(offset);

        assert_eq!(pure_translation.to_matrix(), trans_matrix);
    }

    #[test]
    fn test_identity() {
        let q1 = DualQuaternion::<f64>::identity();
        let q2 = {
            let t = DualQuaternion::from_position([10.0, 13.0, 24.0].into());
            let r = DualQuaternion::from_rotation(Quaternion::from_angle_axis(
                Angle::Degrees(13.0),
                Vector::new([9.0, 2.0, 13.0]).normalized(),
            ));

            t * r
        };

        assert_eq!(q1 * q2, q2);
        assert_eq!(q2 * q1, q2);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        let dual_quaternion = DualQuaternion::new(
            Quaternion::new(1.0, 4.0, 2.0, 1.0),
            Quaternion::new(0.0, 40.0, 23.0, 0.0),
        );

        let dual_quaternion_string = serde_json::to_string(&dual_quaternion).unwrap();
        let dual_quaternion_deserialized = serde_json::from_str(&dual_quaternion_string).unwrap();

        assert_relative_eq!(&dual_quaternion, &dual_quaternion_deserialized);

        let dual_quaternion_data = rmp_serde::to_vec(&dual_quaternion).unwrap();
        let dual_quaternion_deserialized = rmp_serde::from_slice(&dual_quaternion_data).unwrap();

        assert_relative_eq!(&dual_quaternion, &dual_quaternion_deserialized);
    }
}
