// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::utils::arrays::{array_assume_init, array_get_unchecked, array_get_unchecked_mut};
use core::{
    mem::{self, ManuallyDrop, MaybeUninit, offset_of},
    ptr,
};

macro_rules! zip_map_impl {
    ( $func:expr => $($arrays:ident),* $(,)? ) => {{
        let mut result = [const { MaybeUninit::<_>::uninit() }; _];

        for i in 0..(result.len()) {
            unsafe {
                let slot = result.get_unchecked_mut(i);

                $(
                    let $arrays = ptr::read(array_get_unchecked(&$arrays, i));
                )*

                slot.write($func($($arrays),*));
            }
        }

        unsafe { array_assume_init(result) }
    }};
}

macro_rules! zip_impl {
    ([($( $generic:ident ),* $(,)?) ; $num:expr], $($arrays:ident),* $(,)?) => {{
        let mut zipped = [const { MaybeUninit::<($($generic),*)>::uninit() }; $num];

        let mut i = 0;
        while i < N {
            unsafe {
                let slot = array_get_unchecked_mut(&mut zipped, i);

                $(
                    let $arrays = ptr::read(array_get_unchecked(&$arrays, i));
                )*

                slot.write(($( $arrays ),*));
            }
            i += 1;
        }

        let _arrays = ManuallyDrop::new(($($arrays),*));

        unsafe { array_assume_init(zipped) }
    }};
}

macro_rules! unzip_impl {
    ( $array:expr, ( $( $ty:ident : $idx:expr ),* $(,)? ) $(,)? ) => {{
        #[allow(nonstandard_style)]
        let ($(mut $ty),*) = (
            $( [const { MaybeUninit::<$ty>::uninit() }; N] ),*
        );

        let mut i = 0;
        while i < N {
            unsafe {
                macro_rules! tuple {
                    () => { ( $($ty),* ) };
                }

                let slot = $array.as_ptr().add(i);

                $(
                    array_get_unchecked_mut(&mut $ty, i)
                        .write(ptr::read(slot.byte_add(offset_of!(tuple!(), $idx)).cast::<$ty>()));
                )*
            }

            i += 1;
        }

        mem::forget($array);

        #[allow(nonstandard_style)]
        unsafe {
            $(
                let $ty = array_assume_init($ty);
            )*
            ( $($ty),* )
        }
    }};
}

/// Zips two arrays together and applies the function `f` to each memberwise element, returning a fixed
/// size array of the results.
///
/// It is preferred to use this function over chaining the [`zip()`] and [`map()`] methods together (e.g.
/// `zip(array_0, array_1).map(|(x, y)| x + y);`, as it avoids allocating an intermediate array to store the zipped
/// array.
///
/// [`zip()`]: ./fn.zip.html
/// [`map()`]: https://doc.rust-lang.org/stable/std/primitive.array.html#method.map
#[must_use]
#[inline]
pub fn zip_map<T, U, Res, F, const N: usize>(lhs: [T; N], rhs: [U; N], mut f: F) -> [Res; N]
where
    F: FnMut(T, U) -> Res,
{
    zip_map_impl!( f => lhs, rhs, )
}

/// Zips three arrays together and applies the function `f` to each memberwise element, returning a fixed
/// size array of the results.
///
/// It is preferred to use this function over chaining the [`zip()`] and [`map()`] methods together (e.g.
/// `zip3(arr0, arr1, arr2).map(|(x, y, z)| x + y + z);`, as it avoids allocating an intermediate array to store the zipped
/// array.
///
/// [`zip()`]: ./fn.zip.html
/// [`map()`]: https://doc.rust-lang.org/stable/std/primitive.array.html#method.map
#[must_use]
#[inline]
pub fn zip_map3<T, U, V, Res, F, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    mut f: F,
) -> [Res; N]
where
    F: FnMut(T, U, V) -> Res,
{
    zip_map_impl!( f => a0, a1, a2 )
}

/// Zips two fixed-size arrays together, returning a fixed size array of tuples.
///
/// If the result of this function will be immediately used as an intermediate calculation,
/// it is preferred to use the [`zip_map()`] function.
///
/// # Examples
///
/// ```
/// use vectral::utils::zip;
///
/// let nums = [1, 2, 3];
/// let chars = ['a', 'b', 'c'];
///
/// let nums_and_chars: [(i32, char); 3] = zip(nums, chars);
/// assert_eq!(nums_and_chars[0], (1, 'a'));
/// assert_eq!(nums_and_chars[1], (2, 'b'));
/// assert_eq!(nums_and_chars[2], (3, 'c'));
/// ```
///
/// [`zip_map`]: ./fn.zip_map.html
#[doc(alias = "zip2")]
#[must_use]
#[inline(always)]
pub const fn zip<T, U, const N: usize>(lhs: [T; N], rhs: [U; N]) -> [(T, U); N] {
    zip_impl!([(T, U); N], lhs, rhs)
}

/// Zips three fixed-size arrays together, returning a fixed size array of tuples.
///
/// ```
/// use vectral::utils::zip3;
///
/// let nums = [1, 2, 3];
/// let chars = ['a', 'b', 'c'];
/// let bools = [true, false, true];
///
/// let nums_chars_and_bools: [(i32, char, bool); 3] = zip3(nums, chars, bools);
/// assert_eq!(nums_chars_and_bools[0], (1, 'a', true));
/// assert_eq!(nums_chars_and_bools[1], (2, 'b', false));
/// assert_eq!(nums_chars_and_bools[2], (3, 'c', true));
/// ```
///
/// [`zip_map`]: ./fn.zip_map.html
#[must_use]
#[inline(always)]
pub const fn zip3<T, U, V, const N: usize>(a0: [T; N], a1: [U; N], a2: [V; N]) -> [(T, U, V); N] {
    zip_impl!([(T, U, V); N], a0, a1, a2)
}

/// Unzips an array of tuples, returning a tuple of the unzipped elements.
///
/// # Examples
///
/// ```
/// use vectral::utils::arrays::unzip;
///
/// let mixed_data = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')];
///
/// let (numbers, characters) = unzip(mixed_data);
/// assert_eq!(&numbers, &[1, 2, 3, 4]);
/// assert_eq!(&characters, &['a', 'b', 'c', 'd']);
/// ```
#[must_use]
#[inline(always)]
pub const fn unzip<T, U, const N: usize>(array: [(T, U); N]) -> ([T; N], [U; N]) {
    unzip_impl!( array, (T: 0, U: 1) )
}

/// Unzips an array of tuples, returning a tuple of the unzipped elements.
///
/// # Examples
///
/// ```
/// use vectral::utils::arrays::unzip3;
///
/// let mixed_data = [(1, 'a', false), (2, 'b', true), (3, 'c', false), (4, 'd', true)];
///
/// let (numbers, characters, bools) = unzip3(mixed_data);
/// assert_eq!(&numbers, &[1, 2, 3, 4]);
/// assert_eq!(&characters, &['a', 'b', 'c', 'd']);
/// assert_eq!(&bools, &[false, true, false, true]);
/// ```
#[must_use]
#[inline(always)]
pub const fn unzip3<T, U, V, const N: usize>(array: [(T, U, V); N]) -> ([T; N], [U; N], [V; N]) {
    unzip_impl!( array, (T: 0, U: 1, V: 2) )
}
