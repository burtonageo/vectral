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

        mem::forget(($($arrays),*));
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

        unsafe {
            ( $( array_assume_init($ty) ),* )
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
/// # Examples
///
/// ```
/// use vectral::utils::zip_map;
///
/// let array_1 = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"].map(String::from);
/// let array_2 = [1, 2, 3, 4, 5];
///
/// let joined = zip_map(array_1, array_2, |string, num| format!("{}_{}", string, num));
///
/// assert_eq!(&joined, &[
///     "Alpha_1",
///     "Beta_2",
///     "Gamma_3",
///     "Delta_4",
///     "Epsilon_5",
/// ]);
/// ```
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
/// It is preferred to use this function over chaining the [`zip3()`] and [`map()`] methods together (e.g.
/// `zip3(arr0, arr1, arr2).map(|(x, y, z)| x + y + z);`, as it avoids allocating an intermediate array
/// to store the zipped array.
///
/// # Examples
///
/// ```
/// use vectral::utils::zip_map3;
///
/// let array_1 = [0, 1, 2, 3, 4];
/// let array_2 = [1, 2, 3, 4, 5];
/// let array_3 = [2, 3, 4, 5, 6];
///
/// let joined = zip_map3(array_1, array_2, array_3, |n0, n1, n2| n0 + n1 + n2);
///
/// assert_eq!(&joined, &[3, 6, 9, 12, 15]);
/// ```
///
/// [`zip3()`]: ./fn.zip3.html
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

/// Zips four arrays together and applies the function `f` to each memberwise element, returning a fixed
/// size array of the results.
#[must_use]
#[inline]
pub fn zip_map4<T, U, V, W, Res, F, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
    mut f: F,
) -> [Res; N]
where
    F: FnMut(T, U, V, W) -> Res,
{
    zip_map_impl!( f => a0, a1, a2, a3 )
}

/// Zips five arrays together and applies the function `f` to each memberwise element, returning a fixed
/// size array of the results.
#[must_use]
#[inline]
pub fn zip_map5<T, U, V, W, X, Res, F, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
    a4: [X; N],
    mut f: F,
) -> [Res; N]
where
    F: FnMut(T, U, V, W, X) -> Res,
{
    zip_map_impl!( f => a0, a1, a2, a3, a4 )
}

/// Zips six arrays together and applies the function `f` to each memberwise element, returning a fixed
/// size array of the results.
#[must_use]
#[inline]
pub fn zip_map6<T, U, V, W, X, Y, Res, F, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
    a4: [X; N],
    a5: [Y; N],
    mut f: F,
) -> [Res; N]
where
    F: FnMut(T, U, V, W, X, Y) -> Res,
{
    zip_map_impl!( f => a0, a1, a2, a3, a4, a5 )
}

/// Zips seven arrays together and applies the function `f` to each memberwise element, returning a fixed
/// size array of the results.
#[must_use]
#[inline]
pub fn zip_map7<T, U, V, W, X, Y, Z, Res, F, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
    a4: [X; N],
    a5: [Y; N],
    a6: [Z; N],
    mut f: F,
) -> [Res; N]
where
    F: FnMut(T, U, V, W, X, Y, Z) -> Res,
{
    zip_map_impl!( f => a0, a1, a2, a3, a4, a5, a6 )
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

/// Zips four fixed-size arrays together, returning a fixed size array of tuples.
#[must_use]
#[inline(always)]
pub const fn zip4<T, U, V, W, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
) -> [(T, U, V, W); N] {
    zip_impl!([(T, U, V, W); N], a0, a1, a2, a3)
}

/// Zips five fixed-size arrays together, returning a fixed size array of tuples.
#[must_use]
#[inline(always)]
pub const fn zip5<T, U, V, W, X, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
    a4: [X; N],
) -> [(T, U, V, W, X); N] {
    zip_impl!([(T, U, V, W, X); N], a0, a1, a2, a3, a4)
}

/// Zips six fixed-size arrays together, returning a fixed size array of tuples.
#[must_use]
#[inline(always)]
pub const fn zip6<T, U, V, W, X, Y, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
    a4: [X; N],
    a5: [Y; N],
) -> [(T, U, V, W, X, Y); N] {
    zip_impl!([(T, U, V, W, X, Y); N], a0, a1, a2, a3, a4, a5)
}

/// Zips seven fixed-size arrays together, returning a fixed size array of tuples.
#[must_use]
#[inline(always)]
pub const fn zip7<T, U, V, W, X, Y, Z, const N: usize>(
    a0: [T; N],
    a1: [U; N],
    a2: [V; N],
    a3: [W; N],
    a4: [X; N],
    a5: [Y; N],
    a6: [Z; N],
) -> [(T, U, V, W, X, Y, Z); N] {
    zip_impl!([(T, U, V, W, X, Y, Z); N], a0, a1, a2, a3, a4, a5, a6)
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

/// Unzips an array of tuples, returning a tuple of the unzipped elements.
#[must_use]
#[inline(always)]
pub const fn unzip4<T, U, V, W, const N: usize>(
    array: [(T, U, V, W); N],
) -> ([T; N], [U; N], [V; N], [W; N]) {
    unzip_impl!( array, (T: 0, U: 1, V: 2, W: 3) )
}

/// Unzips an array of tuples, returning a tuple of the unzipped elements.
#[must_use]
#[inline(always)]
pub const fn unzip5<T, U, V, W, X, const N: usize>(
    array: [(T, U, V, W, X); N],
) -> ([T; N], [U; N], [V; N], [W; N], [X; N]) {
    unzip_impl!( array, (T: 0, U: 1, V: 2, W: 3, X: 4) )
}

/// Unzips an array of tuples, returning a tuple of the unzipped elements.
#[must_use]
#[inline(always)]
pub const fn unzip6<T, U, V, W, X, Y, const N: usize>(
    array: [(T, U, V, W, X, Y); N],
) -> ([T; N], [U; N], [V; N], [W; N], [X; N], [Y; N]) {
    unzip_impl!( array, (T: 0, U: 1, V: 2, W: 3, X: 4, Y: 5) )
}

/// Unzips an array of tuples, returning a tuple of the unzipped elements.
#[must_use]
#[inline(always)]
pub const fn unzip7<T, U, V, W, X, Y, Z, const N: usize>(
    array: [(T, U, V, W, X, Y, Z); N],
) -> ([T; N], [U; N], [V; N], [W; N], [X; N], [Y; N], [Z; N]) {
    unzip_impl!( array, (T: 0, U: 1, V: 2, W: 3, X: 4, Y: 5, Z: 6) )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zip() {
        let a = [1, 2, 3, 4, 5];
        let b = ['a', 'b', 'c', 'd', 'e'].map(String::from);

        let combined = zip(a, b.clone());

        for (i, (num, string)) in combined.iter().enumerate() {
            let ch = char::from_u32(u32::from(b'a' + i as u8)).unwrap();
            let expected_str = String::from(ch);

            assert_eq!(*num as usize, i + 1);
            assert_eq!(*string, *expected_str);
        }

        let (nums, strs) = unzip(combined);
        assert_eq!(nums, a);
        assert_eq!(strs, b);
    }

    #[test]
    fn test_zip_map() {
        let x = [1, 2, 3, 4, 5].map(|v| format!("{v}"));
        let y = ['a', 'b', 'c', 'd', 'e'].map(String::from);

        let zipped = zip_map(x, y, |x, y| format!("{x}{y}"));

        assert_eq!(&zipped, &["1a", "2b", "3c", "4d", "5e"]);
    }
}
