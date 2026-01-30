// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::utils::num::{ClosedAdd, ClosedMul, One, Zero};
use core::ops::{Add, Mul};

pub mod assertions {
    #[macro_export]
    macro_rules! const_assert_smaller {
        ($x:expr, $y:expr) => {
            #[allow(path_statements)]
            <$crate::utils::assertions::AssertSmaller<$x, $y>>::ASSERT;
        };
    }

    #[macro_export]
    macro_rules! const_assert_larger {
        ($x:expr, $y:expr) => {
            #[allow(path_statements)]
            <$crate::utils::assertions::AssertLarger<$x, $y>>::ASSERT;
        };
    }

    #[macro_export]
    macro_rules! const_assert_larger_or_equal {
        ($x:expr, $y:expr) => {
            #[allow(path_statements)]
            <$crate::utils::assertions::AssertLargerOrEqual<$x, $y>>::ASSERT;
        };
    }

    #[macro_export]
    macro_rules! const_assert_smaller_or_equal {
        ($x:expr, $y:expr) => {
            #[allow(path_statements)]
            <$crate::utils::assertions::AssertSmallerOrEqual<$x, $y>>::ASSERT;
        };
    }

    #[macro_export]
    macro_rules! const_assert_equal {
        ($x:expr, $y:expr) => {
            #[allow(path_statements)]
            <$crate::utils::assertions::AssertEqual<$x, $y>>::ASSERT;
        };
    }

    #[non_exhaustive]
    pub struct AssertSmaller<const X: usize, const Y: usize>;

    impl<const X: usize, const Y: usize> AssertSmaller<{ X }, { Y }> {
        pub const ASSERT: () = assert!(X < Y);
    }

    #[non_exhaustive]
    pub struct AssertLarger<const X: usize, const Y: usize>;

    impl<const X: usize, const Y: usize> AssertLarger<{ X }, { Y }> {
        pub const ASSERT: () = assert!(X > Y);
    }

    #[non_exhaustive]
    pub struct AssertEqual<const X: usize, const Y: usize>;

    impl<const X: usize, const Y: usize> AssertEqual<X, Y> {
        pub const ASSERT: () = assert!(X == Y);
    }

    #[non_exhaustive]
    pub struct AssertSmallerOrEqual<const X: usize, const Y: usize>;

    impl<const X: usize, const Y: usize> AssertSmallerOrEqual<X, Y> {
        pub const ASSERT: () = assert!(X <= Y);
    }

    #[non_exhaustive]
    pub struct AssertLargerOrEqual<const X: usize, const Y: usize>;

    impl<const X: usize, const Y: usize> AssertLargerOrEqual<X, Y> {
        pub const ASSERT: () = assert!(X >= Y);
    }
}

pub mod arrays;
pub mod num;

/// Analogous to the [`Iterator::sum()`] method, but which uses [`Zero`] and [`ClosedAdd`] instead of
/// the [`Sum`] trait.
///
/// [`Iterator::sum()`]: https://doc.rust-lang.org/stable/std/iter/trait.Iterator.html#method.sum
/// [`Sum`]: https://doc.rust-lang.org/stable/std/iter/trait.Sum.html
/// [`ClosedAdd`]: ../trait.ClosedAdd.html
/// [`Zero`]: ../trait.Zero.html
#[must_use]
#[inline(always)]
pub fn sum<I>(iter: I) -> I::Item
where
    I: IntoIterator,
    I::Item: Zero + ClosedAdd,
{
    iter.into_iter().fold(Zero::ZERO, Add::add)
}

/// Analogous to the [`Iterator::product()`] method, but which uses [`One`] and [`ClosedMul`] instead of
/// the [`Product`] trait.
///
/// [`Iterator::product()`]: https://doc.rust-lang.org/stable/std/iter/trait.Iterator.html#method.product
/// [`Product`]: https://doc.rust-lang.org/stable/std/iter/trait.Product.html
/// [`ClosedMul`]: ../trait.ClosedMul.html
/// [`One`]: ../trait.One.html
#[must_use]
#[inline(always)]
pub fn product<I>(iter: I) -> I::Item
where
    I: IntoIterator,
    I::Item: One + ClosedMul,
{
    iter.into_iter().fold(One::ONE, Mul::mul)
}

pub use self::arrays::*;
