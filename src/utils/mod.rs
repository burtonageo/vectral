// SPDX-License-Identifier: MIT OR Apache-2.0

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

#[deprecated = "use the top-level `vectral::num` module"]
pub mod num {
    pub use crate::num::*;
}

pub use self::arrays::*;
