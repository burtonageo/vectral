use crate::utils::num::{
    Bounded, ClosedAdd, ClosedSub, One, Trig, Zero, checked::CheckedAddAssign, n,
};
#[cfg(feature = "serde")]
use core::marker::PhantomData;
use core::{
    cmp::{self, PartialOrd},
    num::NonZeroUsize,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};
use serde_core::de::VariantAccess;
#[cfg(feature = "serde")]
use serde_core::{
    de::{self, Deserialize, Deserializer},
    ser::{Serialize, Serializer},
};

#[derive(Clone, Copy, Debug)]
pub enum Angle<T> {
    Degrees(T),
    Radians(T),
}

impl<T: Zero> Angle<T> {
    #[must_use]
    #[inline]
    pub fn zero() -> Self {
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

impl<T: PartialOrd + Trig> PartialOrd for Angle<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        match (*self, *other) {
            (Angle::Degrees(ref a0), Angle::Degrees(ref a1)) => a0.partial_cmp(a1),
            (a0, a1) => a0.in_radians().partial_cmp(&a1.in_radians()),
        }
    }
}

impl<T: PartialEq + Trig> PartialEq for Angle<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
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
                serializer.serialize_newtype_variant("Angle", 0, "Radians", rads)
            }
        }
    }
}

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
