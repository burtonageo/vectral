#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(incomplete_features)]
#![feature(
    const_index,
    const_trait_impl,
    maybe_uninit_array_assume_init,
    maybe_uninit_uninit_array_transpose,
    generic_const_exprs,
    f16,
    f128
)]

extern crate alloc;

macro_rules! nz {
    ($num:expr) => {
        const {
            match ::core::num::NonZero::new($num) {
                Some(num) => num,
                None => panic!("Given number was equal to 0"),
            }
        }
    };
}

macro_rules! impl_coerce_to_fields {
    (
        $( $type:ident<{ $generic:ident, $const:expr }>  => $fields_type:ident ),+ $(,)?
    ) => {
        $(
            impl<$generic> core::ops::Deref for $type<$generic, $const> {
                type Target = crate::fields:: $fields_type<$generic>;
                #[allow(unsafe_code)]
                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    const _: () = assert!(
                        core::mem::size_of::<$type<u8, $const>>()
                            == core::mem::size_of::<crate::fields:: $fields_type<u8>>()
                    );

                    const _: () = assert!(
                        core::mem::align_of::<$type<u8, $const>>()
                            == core::mem::align_of::<crate::fields:: $fields_type<u8>>()
                    );

                    const _: () = assert!(
                        core::mem::size_of::<$type<f32, $const>>()
                            == core::mem::size_of::<crate::fields:: $fields_type<f32>>()
                    );

                    const _: () = assert!(
                        core::mem::align_of::<$type<f32, $const>>()
                            == core::mem::align_of::<crate::fields:: $fields_type<f32>>()
                    );

                    const _: () = assert!(
                        core::mem::size_of::<$type<f64, $const>>()
                            == core::mem::size_of::<crate::fields:: $fields_type<f64>>()
                    );

                    const _: () = assert!(
                        core::mem::align_of::<$type<f64, $const>>()
                            == core::mem::align_of::<crate::fields:: $fields_type<f64>>()
                    );

                    unsafe { &*(self as *const _ as *const crate::fields:: $fields_type<$generic>) }
                }
            }

            impl<$generic> core::ops::DerefMut for $type<$generic, $const> {
                #[allow(unsafe_code)]
                #[inline(always)]
                fn deref_mut(&mut self) -> &mut Self::Target {
                    unsafe { &mut *(self as *mut _ as *mut crate::fields:: $fields_type<$generic>) }
                }
            }

            impl<$generic> From<$type<$generic, $const>> for crate::fields:: $fields_type<$generic> {
                #[inline]
                fn from(value: $type<$generic, $const>) -> Self {
                    let as_slice: [$generic; $const] = value.into();
                    From::from(as_slice)
                }
            }

            impl<$generic> From<crate::fields:: $fields_type<$generic>> for $type<$generic, $const> {
                #[inline]
                fn from(value: crate::fields:: $fields_type<$generic>) -> Self {
                    let as_slice: [$generic; $const] = value.into();
                    From::from(as_slice)
                }
            }
        )+
    };
}

macro_rules! impl_eq_mint {
    ( $( ( $mint_type:ident, $linalg_type:ident < $type_size:literal > $(,)? ) ),* $(,)? ) => {
        $(
            #[cfg(feature = "mint")]
            impl<T: PartialEq> PartialEq<mint::$mint_type<T>> for $linalg_type<T, $type_size> {
                #[inline]
                fn eq(&self, other: &mint::$mint_type<T>) -> bool {
                    let (lhs, rhs): (&[T; $type_size], &[T; $type_size]) = (self.as_ref(), other.as_ref());
                    PartialEq::eq(lhs, rhs)
                }
            }

            #[cfg(feature = "mint")]
            impl<T: PartialEq> PartialEq<$linalg_type<T, $type_size>> for mint::$mint_type<T> {
                #[inline]
                fn eq(&self, other: &$linalg_type<T, $type_size>) -> bool {
                    let (lhs, rhs): (&[T; $type_size], &[T; $type_size]) = (self.as_ref(), other.as_ref());
                    PartialEq::eq(lhs, rhs)
                }
            }

            #[cfg(all(feature = "approx", feature = "mint"))]
            impl<T: approx::AbsDiffEq> approx::AbsDiffEq<mint::$mint_type<T>> for $linalg_type<T, $type_size>
            where
                T::Epsilon: Copy,
            {
                type Epsilon = T::Epsilon;

                #[inline]
                fn default_epsilon() -> Self::Epsilon {
                    T::default_epsilon()
                }

                #[inline]
                fn abs_diff_eq(&self, other: &mint::$mint_type<T>, epsilon: Self::Epsilon) -> bool {
                    self.iter()
                        .zip(other.as_ref().iter())
                        .all(|(x, y)| x.abs_diff_eq(y, epsilon))
                }
            }

            #[cfg(all(feature = "approx", feature = "mint"))]
            impl<T: approx::AbsDiffEq> approx::AbsDiffEq<$linalg_type<T, $type_size>> for mint::$mint_type<T>
            where
                T::Epsilon: Copy,
            {
                type Epsilon = T::Epsilon;

                #[inline]
                fn default_epsilon() -> Self::Epsilon {
                    T::default_epsilon()
                }

                #[inline]
                fn abs_diff_eq(&self, other: &$linalg_type<T, $type_size>, epsilon: Self::Epsilon) -> bool {
                    self.as_ref()
                        .iter()
                        .zip(other.as_slice())
                        .all(|(x, y)| x.abs_diff_eq(y, epsilon))
                }
            }

            #[cfg(all(feature = "approx", feature = "mint"))]
            impl<T: approx::RelativeEq> approx::RelativeEq<mint::$mint_type<T>> for $linalg_type<T, $type_size>
            where
                T::Epsilon: Copy,
            {
                #[inline]
                fn default_max_relative() -> Self::Epsilon {
                    T::default_max_relative()
                }

                #[inline]
                fn relative_eq(
                    &self,
                    other: &mint::$mint_type<T>,
                    epsilon: Self::Epsilon,
                    max_relative: Self::Epsilon,
                ) -> bool {
                    self.iter()
                        .zip(other.as_ref().iter())
                        .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
                }
            }

            #[cfg(all(feature = "approx", feature = "mint"))]
            impl<T: approx::RelativeEq> approx::RelativeEq<$linalg_type<T, $type_size>> for mint::$mint_type<T>
            where
                T::Epsilon: Copy,
            {
                #[inline]
                fn default_max_relative() -> Self::Epsilon {
                    T::default_max_relative()
                }

                #[inline]
                fn relative_eq(
                    &self,
                    other: &$linalg_type<T, $type_size>,
                    epsilon: Self::Epsilon,
                    max_relative: Self::Epsilon,
                ) -> bool {
                    self.as_ref()
                        .iter()
                        .zip(other.as_slice())
                        .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
                }
            }

            #[cfg(all(feature = "approx", feature = "mint"))]
            impl<T: approx::UlpsEq> approx::UlpsEq<mint::$mint_type<T>> for $linalg_type<T, $type_size>
            where
                T::Epsilon: Copy,
            {
                #[inline]
                fn default_max_ulps() -> u32 {
                    T::default_max_ulps()
                }

                #[inline]
                fn ulps_eq(&self, other: &mint::$mint_type<T>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                    self.iter()
                        .zip(other.as_ref().iter())
                        .all(|(x, y)| x.ulps_eq(y, epsilon, max_ulps))
                }
            }

            #[cfg(all(feature = "approx", feature = "mint"))]
            impl<T: approx::UlpsEq> approx::UlpsEq<$linalg_type<T, $type_size>> for mint::$mint_type<T>
            where
                T::Epsilon: Copy,
            {
                #[inline]
                fn default_max_ulps() -> u32 {
                    T::default_max_ulps()
                }

                #[inline]
                fn ulps_eq(&self, other: &$linalg_type<T, $type_size>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                    self.as_ref()
                        .iter()
                        .zip(other.as_slice())
                        .all(|(x, y)| x.ulps_eq(y, epsilon, max_ulps))
                }
            }
        )*
    };
}

pub mod fields;
pub mod matrix;
pub mod point;
pub mod rotation;
pub mod transform;
pub mod utils;
pub mod vector;
