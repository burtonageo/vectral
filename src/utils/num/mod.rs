// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::utils::num::checked::{CheckedAddAssign, CheckedDiv};
use core::{
    cmp::Ordering,
    num::{
        NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize, NonZeroU8,
        NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize, Saturating, Wrapping,
    },
    ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Shl, ShlAssign, Shr, ShrAssign,
        Sub, SubAssign,
    },
};

pub mod checked;

pub trait ClosedSub: Sized + Sub<Output = Self> {}
pub trait ClosedMul: Sized + Mul<Output = Self> {}
pub trait ClosedAdd: Sized + Add<Output = Self> {}
pub trait ClosedDiv: Sized + Div<Output = Self> {}
pub trait ClosedNeg: Sized + Neg<Output = Self> {}
pub trait ClosedRem: Sized + Rem<Output = Self> {}

impl<T: Sized + Sub<Output = Self>> ClosedSub for T {}
impl<T: Sized + Mul<Output = Self>> ClosedMul for T {}
impl<T: Sized + Add<Output = Self>> ClosedAdd for T {}
impl<T: Sized + Div<Output = Self>> ClosedDiv for T {}
impl<T: Sized + Neg<Output = Self>> ClosedNeg for T {}
impl<T: Sized + Rem<Output = Self>> ClosedRem for T {}

pub trait Scalar:
    Bounded
    + ClosedAdd
    + ClosedDiv
    + ClosedMul
    + ClosedSub
    + AddAssign
    + DivAssign
    + MulAssign
    + SubAssign
    + One
    + PartialEq
    + PartialOrd
    + Zero
{
}

impl<T> Scalar for T where
    T: Bounded
        + ClosedAdd
        + ClosedDiv
        + ClosedMul
        + ClosedSub
        + AddAssign
        + DivAssign
        + MulAssign
        + SubAssign
        + One
        + PartialEq
        + PartialOrd
        + Zero
{
}

pub trait IntScalar:
    Scalar + Shl<Self, Output = Self> + Shr<Self, Output = Self> + ShlAssign + ShrAssign
{
}
impl<T> IntScalar for T where
    T: Scalar + Shl<Self, Output = Self> + Shr<Self, Output = Self> + ShlAssign + ShrAssign
{
}

pub trait Signed: Scalar + Abs + ClosedNeg {
    #[must_use]
    #[inline]
    fn is_negative(self) -> bool {
        self < Self::ZERO
    }

    #[must_use]
    #[inline]
    fn sign(self) -> Self {
        match self.partial_cmp(&Self::ZERO) {
            Some(Ordering::Less) => Self::ONE.neg(),
            Some(Ordering::Greater) => Self::ONE,
            _ => Self::ZERO,
        }
    }
}
impl<T: Scalar + Abs + ClosedNeg> Signed for T {}

pub trait SignedIntScalar: IntScalar + Signed {}
impl<T: IntScalar + Signed> SignedIntScalar for T {}

pub trait Sqrt: Copy {
    #[must_use]
    fn sqrt(self) -> Self;
}

pub trait FloatScalar: Scalar + Trig + Float + Signed {}
impl<T: Scalar + Trig + Float + Signed> FloatScalar for T {}

pub trait Float: Signed + FromFloat + Trig + Sqrt {}
impl<T> Float for T where T: Signed + FromFloat + Trig + Sqrt {}

#[must_use]
#[inline]
pub fn lerp<T, U>(start: T, target: T, t: U) -> T
where
    T: ClosedAdd + Mul<U, Output = T>,
    U: Copy + ClosedSub + One,
{
    let inv_t = U::ONE - t;
    (start * inv_t) + (target * t)
}

pub trait FromFloat<FloatType = f32> {
    #[must_use]
    fn from_float(float_type: FloatType) -> Self;

    #[must_use]
    fn from_float_ceil(float_type: FloatType) -> Self;

    #[must_use]
    fn from_float_floor(float_type: FloatType) -> Self;

    #[must_use]
    fn from_float_round(float_type: FloatType) -> Self;
}

pub trait Abs: Copy {
    #[must_use]
    fn abs(self) -> Self;

    #[must_use]
    fn abs_diff(self, rhs: Self) -> Self;
}

#[doc(alias = "1")]
pub trait One {
    const ONE: Self;
}

#[doc(alias = "0")]
pub trait Zero {
    const ZERO: Self;
}

pub trait Bounded {
    const MIN: Self;
    const MAX: Self;
}

#[must_use]
#[inline]
pub fn n<T: One + CheckedAddAssign + Bounded>(n: NonZeroUsize) -> T {
    let mut val = T::ONE;
    for _ in 0..n.get() - 1 {
        val.checked_add_assign(T::ONE).expect("overflow");
    }
    val
}

#[must_use]
#[inline]
pub fn rat<T: One + CheckedAddAssign + CheckedDiv<Output = T> + Bounded>(
    numerator: NonZeroUsize,
    denominator: NonZeroUsize,
) -> T {
    n::<T>(numerator) / n(denominator)
}
pub trait Trig: Copy {
    #[must_use]
    fn sin(self) -> Self;
    #[must_use]
    fn cos(self) -> Self;
    #[must_use]
    fn tan(self) -> Self;

    #[must_use]
    fn asin(self) -> Self;
    #[must_use]
    fn acos(self) -> Self;
    #[must_use]
    fn atan(self) -> Self;

    #[must_use]
    fn sinh(self) -> Self;
    #[must_use]
    fn asinh(self) -> Self;

    #[must_use]
    fn cosh(self) -> Self;
    #[must_use]
    fn acosh(self) -> Self;

    #[must_use]
    fn tanh(self) -> Self;
    #[must_use]
    fn atanh(self) -> Self;

    #[must_use]
    fn to_radians(self) -> Self;
    #[must_use]
    fn to_degrees(self) -> Self;

    #[must_use]
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
}

macro_rules! fragments {
    ( $( $( #[ $meta:meta] )* $num_ty:ty $(;)? )? ) => {};

    ( $( #[ $meta:meta] )* $num_ty:ty ; zero = $zero:expr $(, $($rest:tt)* )? ) => {

        $( #[ $meta] )*
        impl Zero for $num_ty {
            const ZERO: Self = $zero;
        }

        fragments! { $( #[ $meta] )* $num_ty ; $( $($rest)* )? }
    };

    ( $( #[ $meta:meta] )* $num_ty:ty ; one = $one:expr $(, $($rest:tt)* )? ) => {
        $( #[ $meta] )*
        impl One for $num_ty {
            const ONE: Self = $one;
        }

        fragments! { $( #[ $meta] )* $num_ty ; $( $($rest)* )? }
    };
}

macro_rules! impl_nums {
    (
        $(
            $( #[ $meta:meta] )*
            $num_ty:ty => ( $( $inits:tt )* )
        ),* $(,)?
    ) => {
        $(
            fragments! { $( #[ $meta ] )* $num_ty ; $($inits)* }

            $( #[ $meta] )*
            impl Bounded for $num_ty {
                const MIN: Self = <$num_ty>::MIN;
                const MAX: Self = <$num_ty>::MAX;
            }
        )*
    };
}

impl_nums! {
    u8 => (zero = 0, one = 1),
    u16 => (zero = 0, one = 1),
    u32 => (zero = 0, one = 1),
    u64 => (zero = 0, one = 1),
    u128 => (zero = 0, one = 1),

    i8 => (zero = 0, one = 1),
    i16 => (zero = 0, one = 1),
    i32 => (zero = 0, one = 1),
    i64 => (zero = 0, one = 1),
    i128 => (zero = 0, one = 1),

    #[cfg(feature = "nightly")]
    f16 => (zero = 0.0, one = 1.0),
    f32 => (zero = 0.0, one = 1.0),
    f64 => (zero = 0.0, one = 1.0),
    #[cfg(feature = "nightly")]
    f128 => (zero = 0.0, one = 1.0),

    NonZeroI8 => ( one = NonZeroI8::new(1).unwrap() ),
    NonZeroI16 => ( one = NonZeroI16::new(1).unwrap() ),
    NonZeroI32 => ( one = NonZeroI32::new(1).unwrap() ),
    NonZeroI64 => ( one = NonZeroI64::new(1).unwrap() ),
    NonZeroI128 => ( one = NonZeroI128::new(1).unwrap() ),
    NonZeroIsize => ( one = NonZeroIsize::new(1).unwrap() ),

    NonZeroU8 => ( one = NonZeroU8::new(1).unwrap() ),
    NonZeroU16 => ( one = NonZeroU16::new(1).unwrap() ),
    NonZeroU32 => ( one = NonZeroU32::new(1).unwrap() ),
    NonZeroU64 => ( one = NonZeroU64::new(1).unwrap() ),
    NonZeroU128 => ( one = NonZeroU128::new(1).unwrap() ),
    NonZeroUsize => ( one = NonZeroUsize::new(1).unwrap() ),
}

impl<T: Zero> Zero for Wrapping<T> {
    const ZERO: Self = Wrapping(T::ZERO);
}

impl<T: One> One for Wrapping<T> {
    const ONE: Self = Wrapping(T::ONE);
}

impl<T: Bounded> Bounded for Wrapping<T> {
    const MIN: Self = Wrapping(T::MIN);
    const MAX: Self = Wrapping(T::MAX);
}

impl<T: Zero> Zero for Saturating<T> {
    const ZERO: Self = Saturating(T::ZERO);
}

impl<T: One> One for Saturating<T> {
    const ONE: Self = Saturating(T::ONE);
}

impl<T: Bounded> Bounded for Saturating<T> {
    const MIN: Self = Saturating(T::MIN);
    const MAX: Self = Saturating(T::MAX);
}

impl<T: Zero, const N: usize> Zero for [T; N] {
    const ZERO: Self = [T::ZERO; N];
}

macro_rules! impl_from_float_type {
    ($impl_type:ty, $from_ty:ty) => {
        #[cfg(feature = "std")]
        impl FromFloat<$from_ty> for $impl_type {
            #[inline]
            fn from_float(float_type: $from_ty) -> Self {
                float_type as $impl_type
            }

            #[inline]
            fn from_float_ceil(float_type: $from_ty) -> Self {
                (float_type.ceil()) as $impl_type
            }

            #[inline]
            fn from_float_floor(float_type: $from_ty) -> Self {
                (float_type.floor()) as $impl_type
            }

            #[inline]
            fn from_float_round(float_type: $from_ty) -> Self {
                (float_type.round()) as $impl_type
            }
        }

        #[cfg(all(feature = "libm", not(feature = "std")))]
        impl FromFloat<$from_ty> for $impl_type {
            #[inline]
            fn from_float(float_type: $from_ty) -> Self {
                float_type as $impl_type
            }

            #[inline]
            fn from_float_ceil(float_type: $from_ty) -> Self {
                libm::Libm::<$from_ty>::ceil(float_type) as $impl_type
            }

            #[inline]
            fn from_float_floor(float_type: $from_ty) -> Self {
                libm::Libm::<$from_ty>::floor(float_type) as $impl_type
            }

            #[inline]
            fn from_float_round(float_type: $from_ty) -> Self {
                libm::Libm::<$from_ty>::round(float_type) as $impl_type
            }
        }
    };
}

macro_rules! impl_float_traits {
    ( $($type:ty),* $(,)? ) => {
        $(
            #[cfg(feature = "std")]
            impl Sqrt for $type {
                #[inline(always)]
                fn sqrt(self) -> Self {
                    <$type>::sqrt(self)
                }
            }

            #[cfg(all(feature = "libm", not(feature = "std")))]
            impl Sqrt for $type {
                #[inline(always)]
                fn sqrt(self) -> Self {
                    libm::Libm::<$type>::sqrt(self)
                }
            }

            #[cfg(feature = "std")]
            impl Trig for $type {
                #[inline]
                fn sin(self) -> Self  {
                    <$type>::sin(self)
                }

                #[inline]
                fn cos(self) -> Self  {
                    <$type>::cos(self)
                }

                #[inline]
                fn tan(self) -> Self  {
                    <$type>::tan(self)
                }

                #[inline]
                fn asin(self) -> Self  {
                    <$type>::asin(self)
                }

                #[inline]
                fn acos(self) -> Self  {
                    <$type>::acos(self)
                }

                #[inline]
                fn atan(self) -> Self  {
                    <$type>::atan(self)
                }

                #[inline]
                fn sinh(self) -> Self  {
                    <$type>::sinh(self)
                }

                #[inline]
                fn asinh(self) -> Self  {
                    <$type>::asinh(self)
                }

                #[inline]
                fn cosh(self) -> Self  {
                    <$type>::cosh(self)
                }

                #[inline]
                fn acosh(self) -> Self  {
                    <$type>::acosh(self)
                }

                #[inline]
                fn tanh(self) -> Self  {
                    <$type>::tanh(self)
                }

                #[inline]
                fn atanh(self) -> Self  {
                    <$type>::atanh(self)
                }

                #[inline]
                fn to_radians(self) -> Self {
                    <$type>::to_radians(self)
                }

                #[inline]
                fn to_degrees(self) -> Self {
                    <$type>::to_degrees(self)
                }

                #[inline]
                fn sin_cos(self) -> (Self, Self) {
                    <$type>::sin_cos(self)
                }
            }

            #[cfg(all(feature = "libm", not(feature = "std")))]
            impl Trig for $type {
                #[inline]
                fn sin(self) -> Self  {
                    libm::Libm::<$type>::sin(self)
                }

                #[inline]
                fn cos(self) -> Self  {
                    libm::Libm::<$type>::cos(self)
                }

                #[inline]
                fn tan(self) -> Self  {
                    libm::Libm::<$type>::tan(self)
                }

                #[inline]
                fn asin(self) -> Self  {
                    libm::Libm::<$type>::asin(self)
                }

                #[inline]
                fn acos(self) -> Self  {
                    libm::Libm::<$type>::acos(self)
                }

                #[inline]
                fn atan(self) -> Self  {
                    libm::Libm::<$type>::atan(self)
                }

                #[inline]
                fn sinh(self) -> Self  {
                    libm::Libm::<$type>::sinh(self)
                }

                #[inline]
                fn asinh(self) -> Self  {
                    libm::Libm::<$type>::asinh(self)
                }

                #[inline]
                fn cosh(self) -> Self  {
                    libm::Libm::<$type>::cosh(self)
                }

                #[inline]
                fn acosh(self) -> Self  {
                    libm::Libm::<$type>::acosh(self)
                }

                #[inline]
                fn tanh(self) -> Self  {
                    libm::Libm::<$type>::tanh(self)
                }

                #[inline]
                fn atanh(self) -> Self  {
                    libm::Libm::<$type>::atanh(self)
                }

                #[inline]
                fn to_radians(self) -> Self {
                    <$type>::to_radians(self)
                }

                #[inline]
                fn to_degrees(self) -> Self {
                    <$type>::to_degrees(self)
                }
            }


            #[cfg(all(feature = "std", feature = "nightly", not(feature = "libm")))]
            impl_from_float_type!($type, f16);
            #[cfg(all(feature = "std", feature = "nightly", not(feature = "libm")))]
            impl_from_float_type!($type, f128);

            impl_from_float_type!($type, f32);
            impl_from_float_type!($type, f64);
        )*
    };
}

impl_float_traits! {
    f32, f64,
}

#[cfg(all(feature = "std", feature = "nightly", not(feature = "libm")))]
impl_float_traits! {
    f16, f128,
}

macro_rules! impl_abs_for_signed_types {
    (
        $($ty:ty),* $(,)?
    ) => {
        $(
            impl Abs for $ty {
                #[inline]
                fn abs(self) -> Self {
                    <$ty>::abs(self)
                }

                #[inline]
                fn abs_diff(self, rhs: Self) -> Self {
                    (self - rhs).abs()
                }
            }
        )*
    };
}

impl_abs_for_signed_types! {
    i8, i16, i32, i64, i128, isize,
    f32, f64,
}

#[cfg(feature = "nightly")]
impl_abs_for_signed_types! {
    f16, f128,
}

macro_rules! impl_abs_for_unsigned_types {
    (
        $($ty:ty),* $(,)?
    ) => {
        $(
            impl Abs for $ty {
                #[inline]
                fn abs(self) -> Self {
                    self
                }

                #[inline]
                fn abs_diff(self, rhs: Self) -> Self {
                    if self > rhs {
                        self - rhs
                    } else {
                        rhs - self
                    }
                }
            }
        )*
    };
}

impl_abs_for_unsigned_types! {
    u8, u16, u32, u64, u128, usize,
}
