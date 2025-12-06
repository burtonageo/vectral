use crate::utils::num::Zero;

macro_rules! decl_fields {
    (
        $(
            $ty_name:ident <{ $dim:expr }> {
                $( $field:ident ),+
                $(,)?
            }
        )*
    ) => {
        $(
            #[repr(C)]
            #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
            pub struct $ty_name <T> {
                $( pub $field : T ),+
            }

            impl<T> $ty_name<T> {
                #[must_use]
                #[inline]
                pub const fn new( $( $field : T ),+ ) -> Self {
                    Self {
                        $($field),+
                    }
                }
            }

            impl<T: Copy> $ty_name<T> {
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
            unsafe impl<T: bytemuck::Pod> bytemuck::Pod for $ty_name<T>{}
        )*
    };
}

decl_fields! {
    X <{ 1 }> { x }
    Xy <{ 2 }> { x, y }
    Xyz <{ 3 }> { x, y, z}
    Xyzw <{ 4 }> { x, y, z, w, }
}
