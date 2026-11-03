// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::num::{Bounded, ClosedAdd, ClosedSub, One, Trig, Zero, checked::CheckedAddAssign, n};
#[cfg(feature = "serde")]
use core::marker::PhantomData;
use core::{
    cmp::{self, PartialOrd},
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
#[cfg(feature = "serde")]
use serde_core::{
    de::{self, Deserialize, Deserializer, VariantAccess},
    ser::{Serialize, Serializer},
};

pub use Angle::Degrees;
pub use Angle::Radians;

#[derive(Clone, Copy, Debug)]
pub enum Angle<T> {
    Degrees(T),
    Radians(T),
}

impl<T: Zero> Angle<T> {
    #[must_use]
    #[inline]
    pub const fn zero() -> Self {
        Self::Degrees(T::ZERO)
    }
}

macro_rules! nz_degs {
    ($n:expr) => {
        n(const { NonZeroUsize::new($n).expect("not zero") })
    };
}

impl<T: One + CheckedAddAssign + Bounded> Angle<T> {
    #[must_use]
    #[inline]
    pub fn quarter() -> Self {
        Self::Degrees(nz_degs!(90))
    }

    #[must_use]
    #[inline]
    pub fn half() -> Self {
        Self::Degrees(nz_degs!(180))
    }

    #[must_use]
    #[inline]
    pub fn three_quarters() -> Self {
        Self::Degrees(nz_degs!(270))
    }

    #[must_use]
    #[inline]
    pub fn full() -> Self {
        Self::Degrees(nz_degs!(360))
    }
}

impl<T: Zero + One + CheckedAddAssign + Bounded + PartialOrd> Angle<T>
where
    Angle<T>: PartialOrd,
{
    /// Clamps the given `Angle` to within a full clockwise circle (From 0º to 360º).
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::rotation::angle::Angle;
    ///
    /// let within_circle = Angle::Degrees(270.0);
    /// assert_eq!(within_circle.clamped_to_positive_circle(), Angle::Degrees(270.0));
    ///
    /// let outside_circle = Angle::Degrees(450.0);
    /// assert_eq!(outside_circle.clamped_to_positive_circle(), Angle::full());
    ///
    /// let within_neg_circle = Angle::Degrees(-230.0);
    /// assert_eq!(within_neg_circle.clamped_to_positive_circle(), Angle::zero());
    /// ```
    #[inline]
    pub fn clamped_to_positive_circle(self) -> Self {
        if self < Self::zero() {
            Angle::ZERO
        } else {
            let full = Self::full();
            if self > full { full } else { self }
        }
    }
}

impl<T: Neg<Output = T> + Clone + One + CheckedAddAssign + Bounded + PartialOrd> Angle<T>
where
    Angle<T>: PartialOrd,
{
    /// Clamps the given `Angle` to within a full circle both clockwise
    /// and counterclockwise (From -360º to 360º).
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::rotation::angle::Angle;
    ///
    /// let within_circle = Angle::Degrees(270.0);
    /// assert_eq!(within_circle.clamped_to_circle(), Angle::Degrees(270.0));
    ///
    /// let outside_circle = Angle::Degrees(450.0);
    /// assert_eq!(outside_circle.clamped_to_circle(), Angle::full());
    ///
    /// let within_neg_circle = Angle::Degrees(-230.);
    /// assert_eq!(within_neg_circle.clamped_to_circle(), Angle::Degrees(-230.0));
    /// 
    /// let outside_neg_circle = Angle::Degrees(-1230.);
    /// assert_eq!(outside_neg_circle.clamped_to_circle(), -Angle::<f64>::full());
    /// ```
    #[inline]
    pub fn clamped_to_circle(self) -> Self {
        let full = Self::full();
        let neg_full = full.clone().neg();

        if self < neg_full {
            neg_full
        } else if self > full {
            full
        } else {
            self
        }
    }
}

impl<T: Neg> Neg for Angle<T> {
    type Output = Angle<T::Output>;
    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Self::Degrees(ang) => Angle::Degrees(ang.neg()),
            Self::Radians(ang) => Angle::Radians(ang.neg()),
        }
    }
}

impl<T: Div<U>, U> Div<U> for Angle<T> {
    type Output = Angle<<T as Div<U>>::Output>;
    #[inline]
    fn div(self, rhs: U) -> Self::Output {
        match self {
            Self::Degrees(ang) => Angle::Degrees(ang.div(rhs)),
            Self::Radians(ang) => Angle::Radians(ang.div(rhs)),
        }
    }
}

impl<T: DivAssign<U>, U> DivAssign<U> for Angle<T> {
    #[inline]
    fn div_assign(&mut self, rhs: U) {
        match *self {
            Self::Degrees(ref mut ang) => ang.div_assign(rhs),
            Self::Radians(ref mut ang) => ang.div_assign(rhs),
        }
    }
}

impl<T: Mul<U>, U> Mul<U> for Angle<T> {
    type Output = Angle<<T as Mul<U>>::Output>;
    #[inline]
    fn mul(self, rhs: U) -> Self::Output {
        match self {
            Self::Degrees(ang) => Angle::Degrees(ang.mul(rhs)),
            Self::Radians(ang) => Angle::Radians(ang.mul(rhs)),
        }
    }
}

impl<T: MulAssign<U>, U> MulAssign<U> for Angle<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: U) {
        match *self {
            Self::Degrees(ref mut ang) => ang.mul_assign(rhs),
            Self::Radians(ref mut ang) => ang.mul_assign(rhs),
        }
    }
}

impl<T: Trig + ClosedAdd> Add for Angle<T> {
    type Output = Angle<T>;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Degrees(lhs), Self::Degrees(rhs)) => Self::Degrees(lhs + rhs),
            (lhs, rhs) => Self::Radians(lhs.in_radians() + rhs.in_radians()),
        }
    }
}

impl<T: Trig + AddAssign> AddAssign for Angle<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        match *self {
            Self::Degrees(ref mut lhs) => lhs.add_assign(rhs.in_degrees()),
            Self::Radians(ref mut lhs) => lhs.add_assign(rhs.in_radians()),
        }
    }
}

impl<T: Trig + ClosedSub> Sub for Angle<T> {
    type Output = Angle<T>;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Degrees(lhs), Self::Degrees(rhs)) => Self::Degrees(lhs - rhs),
            (lhs, rhs) => Self::Radians(lhs.in_radians() - rhs.in_radians()),
        }
    }
}

impl<T: Trig + SubAssign> SubAssign for Angle<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        match *self {
            Self::Degrees(ref mut lhs) => lhs.sub_assign(rhs.in_degrees()),
            Self::Radians(ref mut lhs) => lhs.sub_assign(rhs.in_radians()),
        }
    }
}

impl<T: PartialOrd<U> + Trig, U: Trig> PartialOrd<Angle<U>> for Angle<T> {
    #[inline]
    fn partial_cmp(&self, other: &Angle<U>) -> Option<cmp::Ordering> {
        match (*self, *other) {
            (Angle::Degrees(ref a0), Angle::Degrees(ref a1)) => a0.partial_cmp(a1),
            (a0, a1) => a0.in_radians().partial_cmp(&a1.in_radians()),
        }
    }
}

impl<T: Ord + Trig> Ord for Angle<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match (*self, *other) {
            (Angle::Degrees(ref a0), Angle::Degrees(ref a1)) => a0.cmp(a1),
            (a0, a1) => a0.in_radians().cmp(&a1.in_radians()),
        }
    }
}

impl<T: PartialEq<U> + Trig, U: Trig> PartialEq<Angle<U>> for Angle<T> {
    #[inline]
    fn eq(&self, other: &Angle<U>) -> bool {
        match (*self, *other) {
            (Angle::Degrees(ref a0), Angle::Degrees(ref a1)) => a0.eq(a1),
            (a0, a1) => a0.in_radians().eq(&a1.in_radians()),
        }
    }
}

impl<T: Eq + Trig> Eq for Angle<T> {}

impl<T: Default> Default for Angle<T> {
    #[inline]
    fn default() -> Self {
        Self::Radians(Default::default())
    }
}

impl<T: Zero> Zero for Angle<T> {
    const ZERO: Self = Self::Degrees(Zero::ZERO);
}

impl<T: Trig> Angle<T> {
    #[must_use]
    #[inline]
    pub fn in_degrees(self) -> T {
        match self {
            Self::Degrees(degrees) => degrees,
            Self::Radians(radians) => radians.to_degrees(),
        }
    }

    #[must_use]
    #[inline]
    pub fn in_radians(self) -> T {
        match self {
            Self::Degrees(degrees) => degrees.to_radians(),
            Self::Radians(radians) => radians,
        }
    }
}

impl<T: Trig> Trig for Angle<T> {
    #[inline]
    fn sin(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().sin()),
            Self::Radians(radians) => Angle::Radians(radians.sin()),
        }
    }

    #[inline]
    fn cos(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().cos()),
            Self::Radians(radians) => Angle::Radians(radians.cos()),
        }
    }

    #[inline]
    fn tan(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().tan()),
            Self::Radians(radians) => Angle::Radians(radians.tan()),
        }
    }

    #[inline]
    fn asin(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().asin()),
            Self::Radians(radians) => Angle::Radians(radians.asin()),
        }
    }

    #[inline]
    fn acos(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().acos()),
            Self::Radians(radians) => Angle::Radians(radians.acos()),
        }
    }

    #[inline]
    fn atan(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().atan()),
            Self::Radians(radians) => Angle::Radians(radians.atan()),
        }
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Angle::Radians(self.in_radians().atan2(other.in_radians()))
    }

    #[inline]
    fn sinh(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().sinh()),
            Self::Radians(radians) => Angle::Radians(radians.sinh()),
        }
    }

    #[inline]
    fn asinh(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().asinh()),
            Self::Radians(radians) => Angle::Radians(radians.asinh()),
        }
    }

    #[inline]
    fn cosh(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().cosh()),
            Self::Radians(radians) => Angle::Radians(radians.cosh()),
        }
    }

    #[inline]
    fn acosh(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().sin()),
            Self::Radians(radians) => Angle::Radians(radians.sin()),
        }
    }

    #[inline]
    fn tanh(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().sin()),
            Self::Radians(radians) => Angle::Radians(radians.sin()),
        }
    }

    #[inline]
    fn atanh(self) -> Self {
        match self {
            Self::Degrees(degrees) => Angle::Radians(degrees.to_radians().sin()),
            Self::Radians(radians) => Angle::Radians(radians.sin()),
        }
    }

    #[inline]
    fn to_radians(self) -> Self {
        Self::Radians(self.in_radians())
    }

    #[inline]
    fn to_degrees(self) -> Self {
        Self::Degrees(self.in_degrees())
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for Angle<T> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match *self {
            Self::Degrees(ref degs) => {
                serializer.serialize_newtype_variant("Angle", 0, "Degrees", degs)
            }
            Self::Radians(ref rads) => {
                serializer.serialize_newtype_variant("Angle", 1, "Radians", rads)
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Angle<T> {
    #[inline]
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AngleVisitor<T>(PhantomData<Angle<T>>);

        impl<'de, T: Deserialize<'de>> de::Visitor<'de> for AngleVisitor<T> {
            type Value = Angle<T>;
            #[inline]
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("angle of Degrees or Radians")
            }

            #[inline]
            fn visit_enum<A: de::EnumAccess<'de>>(self, data: A) -> Result<Self::Value, A::Error> {
                enum AngleUnit {
                    Degrees,
                    Radians,
                }

                impl<'de> Deserialize<'de> for AngleUnit {
                    #[inline]
                    fn deserialize<D: Deserializer<'de>>(
                        deserializer: D,
                    ) -> Result<Self, D::Error> {
                        struct Visitor;
                        impl<'de> de::Visitor<'de> for Visitor {
                            type Value = AngleUnit;
                            #[inline]
                            fn expecting(
                                &self,
                                formatter: &mut std::fmt::Formatter,
                            ) -> std::fmt::Result {
                                formatter.write_str("`Degrees` or `Angles`")
                            }

                            #[inline]
                            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                                match v {
                                    "Degrees" => Ok(AngleUnit::Degrees),
                                    "Radians" => Ok(AngleUnit::Radians),
                                    _ => {
                                        Err(de::Error::invalid_value(de::Unexpected::Str(v), &self))
                                    }
                                }
                            }

                            #[inline]
                            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                                match v {
                                    0 => Ok(AngleUnit::Degrees),
                                    1 => Ok(AngleUnit::Radians),
                                    _ => Err(de::Error::invalid_value(
                                        de::Unexpected::Signed(v),
                                        &self,
                                    )),
                                }
                            }

                            #[inline]
                            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                                match v {
                                    0 => Ok(AngleUnit::Degrees),
                                    1 => Ok(AngleUnit::Radians),
                                    _ => Err(de::Error::invalid_value(
                                        de::Unexpected::Unsigned(v),
                                        &self,
                                    )),
                                }
                            }
                        }

                        deserializer.deserialize_identifier(Visitor)
                    }
                }

                let (angle_unit, variant_data) = data.variant::<AngleUnit>()?;
                let data = variant_data.newtype_variant::<T>()?;
                match angle_unit {
                    AngleUnit::Degrees => Ok(Angle::Degrees(data)),
                    AngleUnit::Radians => Ok(Angle::Radians(data)),
                }
            }
        }

        const VARIANTS: &'_ [&'_ str] = &["Degrees", "Radians"];
        deserializer.deserialize_enum("Angle", VARIANTS, AngleVisitor(PhantomData))
    }
}

#[cfg(feature = "approx")]
impl<T: approx::AbsDiffEq<U> + Trig, U: Trig> approx::AbsDiffEq<Angle<U>> for Angle<T> {
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Angle<U>, epsilon: Self::Epsilon) -> bool {
        match (*self, *other) {
            (Angle::Degrees(ref d0), Angle::Degrees(ref d1)) => d0.abs_diff_eq(d1, epsilon),
            (lhs, rhs) => lhs.in_radians().abs_diff_eq(&rhs.in_radians(), epsilon),
        }
    }
}

#[cfg(feature = "approx")]
impl<T: approx::RelativeEq<U> + Trig, U: Trig> approx::RelativeEq<Angle<U>> for Angle<T> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Angle<U>,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        match (*self, *other) {
            (Angle::Degrees(ref d0), Angle::Degrees(ref d1)) => {
                d0.relative_eq(d1, epsilon, max_relative)
            }
            (lhs, rhs) => lhs
                .in_radians()
                .relative_eq(&rhs.in_radians(), epsilon, max_relative),
        }
    }
}

#[cfg(feature = "approx")]
impl<T: approx::UlpsEq<U> + Trig, U: Trig> approx::UlpsEq<Angle<U>> for Angle<T> {
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Angle<U>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        match (*self, *other) {
            (Angle::Degrees(ref d0), Angle::Degrees(ref d1)) => d0.ulps_eq(d1, epsilon, max_ulps),
            (lhs, rhs) => lhs
                .in_radians()
                .ulps_eq(&rhs.in_radians(), epsilon, max_ulps),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_clamp() {
        let bigger_than_circle = Angle::Degrees(720.0);
        let within_circle = Angle::Degrees(138.0);
        let within_neg_circle = Angle::Degrees(-68.0);
        let smaller_than_neg_circle = Angle::Degrees(-1283.0);

        assert_eq!(bigger_than_circle.clamped_to_positive_circle(), Angle::<f32>::full());
        assert_eq!(within_circle.clamped_to_positive_circle(), within_circle);
        assert_eq!(within_neg_circle.clamped_to_positive_circle(), Angle::<f32>::ZERO);
        assert_eq!(smaller_than_neg_circle.clamped_to_positive_circle(), Angle::<f32>::ZERO);

        assert_eq!(bigger_than_circle.clamped_to_circle(), Angle::<f32>::full());
        assert_eq!(within_circle.clamped_to_circle(), within_circle);
        assert_eq!(within_neg_circle.clamped_to_circle(), within_neg_circle);
        assert_eq!(smaller_than_neg_circle.clamped_to_circle(), Angle::<f32>::full().neg());
    }
}

