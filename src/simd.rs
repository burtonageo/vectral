// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    rotation::angle::Angle,
    utils::num::{ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub, Sqrt, Trig, Zero},
    vector::Vector,
};
use core::{
    borrow::{Borrow, BorrowMut},
    convert::{AsMut, AsRef},
    ops::{Add, Deref, DerefMut, Div, Mul, Sub},
    simd::{Simd, SimdElement},
};
#[cfg(feature = "serde")]
use serde_core::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, Serializer},
};
use std::ops::DivAssign;

pub trait SimdMul<U = Self>: Copy {
    type Output;
    #[must_use]
    fn simd_mul(self, rhs: U) -> Self::Output;
}

pub trait SimdDiv<U = Self>: Copy {
    type Output;
    #[must_use]
    fn simd_div(self, rhs: U) -> Self::Output;
}

pub trait SimdAdd<U = Self>: Copy {
    type Output;
    #[must_use]
    fn simd_add(self, rhs: U) -> Self::Output;
}

pub trait SimdSub<U = Self>: Copy {
    type Output;
    #[must_use]
    fn simd_sub(self, rhs: U) -> Self::Output;
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SimdValue<T>(pub T);

impl<T> AsRef<T> for SimdValue<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T> AsMut<T> for SimdValue<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> Borrow<T> for SimdValue<T> {
    #[inline]
    fn borrow(&self) -> &T {
        &self.0
    }
}

impl<T> BorrowMut<T> for SimdValue<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> Deref for SimdValue<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for SimdValue<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<T> for SimdValue<T> {
    #[inline]
    fn from(value: T) -> Self {
        SimdValue(value)
    }
}

impl<T: SimdMul<U>, U> Mul<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn mul(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_mul(rhs.0))
    }
}

impl<T: SimdDiv<U>, U> Div<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn div(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_div(rhs.0))
    }
}

impl<T: SimdAdd<U>, U> Add<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn add(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_add(rhs.0))
    }
}

impl<T: SimdSub<U>, U> Sub<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn sub(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_sub(rhs.0))
    }
}

impl<T: SimdMul<U>, U> SimdMul<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn simd_mul(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_mul(rhs.0))
    }
}

impl<T: SimdDiv<U>, U> SimdDiv<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn simd_div(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_div(rhs.0))
    }
}

impl<T: SimdAdd<U>, U> SimdAdd<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn simd_add(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_add(rhs.0))
    }
}

impl<T: SimdSub<U>, U> SimdSub<SimdValue<U>> for SimdValue<T> {
    type Output = SimdValue<T::Output>;
    #[inline]
    fn simd_sub(self, rhs: SimdValue<U>) -> Self::Output {
        SimdValue(self.0.simd_sub(rhs.0))
    }
}

impl<T: SimdElement + ClosedAdd + Zero, const N: usize> SimdValue<Vector<T, N>>
where
    Simd<T, N>: ClosedMul,
{
    #[must_use]
    #[inline]
    pub fn dot<I: Into<SimdValue<Vector<T, N>>>>(self, rhs: I) -> T {
        self.0.simd_dot(rhs)
    }

    #[must_use]
    #[inline]
    pub fn len_squared(self) -> T {
        self.0.simd_len_squared()
    }
}

impl<T: SimdElement + ClosedAdd + Zero + Sqrt, const N: usize> SimdValue<Vector<T, N>>
where
    Simd<T, N>: ClosedMul,
{
    #[must_use]
    #[inline]
    pub fn len(self) -> T {
        self.0.simd_len().sqrt()
    }
}

impl<T: SimdElement + ClosedAdd + Zero + Sqrt + Div, const N: usize> SimdValue<Vector<T, N>>
where
    Simd<T, N>: ClosedMul + DivAssign,
{
    #[inline]
    pub fn normalize(&mut self) {
        let len = self.len();
        let mut simd = Simd::from_array(self.to_array());
        simd /= Simd::splat(len);
        *self = SimdValue(From::from(simd));
    }

    #[must_use]
    #[inline]
    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }
}

impl<T: SimdElement + ClosedNeg> SimdValue<Vector<T>>
where
    Simd<T, 3>: ClosedMul + ClosedSub,
{
    #[must_use]
    #[inline]
    pub fn cross<I: Into<SimdValue<Vector<T>>>>(self, rhs: I) -> Self {
        SimdValue(self.0.simd_cross(rhs))
    }
}

impl<T, const N: usize> SimdValue<Vector<T, N>>
where
    T: SimdElement + ClosedAdd + ClosedMul + ClosedDiv + Sqrt + Trig + Zero,
    Simd<T, N>: ClosedMul,
{
    #[must_use]
    #[inline]
    pub fn angle_between(self, other: Self) -> Angle<T> {
        let a = self;
        let b = other;
        Angle::Radians(T::acos(a.dot(b) / a.len() * b.len()))
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize> Serialize for SimdValue<T> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for SimdValue<T> {
    #[inline]
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(SimdValue)
    }
}
