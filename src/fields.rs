// SPDX-License-Identifier: MIT OR Apache-2.0

//! Types which allow struct-like access to the fields of [`Point`]s and [`Vector`]s
//!
//! [`Vector`]: ../vector/struct.Vector.html
//! [`Point`]: ../point/struct.Point.html

use crate::utils::num::Zero;

macro_rules! decl_fields {
    (
        $(
            $( #[ $meta:meta ] )*
            $ty_name:ident <{ $dim:expr }> {
                $(
                    $( #[ $field_meta:meta ] )*
                    $field:ident
                ),+
                $(,)?
            }
        )*
    ) => {
        $(
            $( #[ $meta ] )*
            #[repr(C)]
            #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
            pub struct $ty_name <T> {
                $(
                    $( #[ $field_meta ] )*
                    pub $field : T
                ),+
            }

            impl<T> $ty_name<T> {
                #[doc = concat!("Create a new `", stringify!($ty_name), "` from the given values.")]
                #[must_use]
                #[inline]
                pub const fn new( $( $field : T ),+ ) -> Self {
                    Self {
                        $($field),+
                    }
                }
            }

            impl<T: Copy> $ty_name<T> {
                #[doc = concat!(
                    "Create a new `", stringify!($ty_name), "` where each element in the `",
                    stringify!($ty_name), "` is set to the given `value`."
                )]
                #[must_use]
                #[inline]
                pub const fn splat(value: T) -> Self {
                    Self {
                        $( $field: value ),+
                    }
                }
            }

            impl<T: Zero> Zero for $ty_name<T> {
                const ZERO: Self = Self {
                    $( $field : Zero::ZERO ),+
                };
            }

            impl<T> From<$ty_name<T>> for [T; $dim] {
                #[inline]
                fn from($ty_name { $( $field ),+}: $ty_name<T>) -> Self {
                    [ $($field),+ ]
                }
            }

            impl<T> From<[T; $dim]> for $ty_name<T> {
                #[inline]
                fn from([ $( $field ),+ ]: [T ; $dim]) -> Self {
                    Self {
                        $($field),+
                    }
                }
            }

            #[cfg(feature = "bytemuck")]
            unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for $ty_name<T> {
                #[inline]
                fn zeroed() -> Self {
                    Self {
                        $( $field: bytemuck::Zeroable::zeroed() ),+
                    }
                }
            }

            #[cfg(feature = "bytemuck")]
            unsafe impl<T: bytemuck::Pod> bytemuck::Pod for $ty_name<T> {}
        )*
    };
}

decl_fields! {
    /// A field with a single scalar value representing a single axis.
    X <{ 1 }> {
        /// The value representing the x-axis value.
        x,
    }
    /// A field with a two scalar values representing a two dimensional space.
    Xy <{ 2 }> {
        /// The value representing the x-axis value.
        x,
        /// The value representing the y-axis value.
        y,
    }
    /// A field with a three scalar values representing a three dimensional space.
    Xyz <{ 3 }> {
        /// The value representing the x-axis value.
        x,
        /// The value representing the y-axis value.
        y,
        /// The value representing the z-axis value.
        z,
    }
    /// A field with a four scalar values representing a four dimensional space.
    Xyzw <{ 4 }> {
        /// The value representing the x-axis value.
        x,
        /// The value representing the y-axis value.
        y,
        /// The value representing the z-axis value.
        z,
        /// The value representing the w-axis value.
        w,
    }
}
