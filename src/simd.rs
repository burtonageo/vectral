
// SPDX-License-Identifier: MIT OR Apache-2.0

use core::{
    borrow::{Borrow, BorrowMut},
    convert::{AsMut, AsRef},
    ops::{Add, Deref, DerefMut, Div, Mul, Sub},
};

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

#[derive(Clone, Copy, Debug)]
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
