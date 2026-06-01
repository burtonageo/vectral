// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{const_assert_larger_or_equal, const_assert_smaller, const_assert_smaller_or_equal};
use core::{
    marker::PhantomData,
    mem::{self, MaybeUninit},
    ptr, slice,
};

mod zip;

pub use self::zip::{unzip, unzip3, zip, zip_map, zip_map3, zip3};

/// Shrinks an array, returning a new array with `NEW_LEN` elements.
///
/// # Examples
///
/// ```
/// # use vectral::utils::shrink_to;
/// let array = [1, 2, 3, 4, 5];
/// let array: [i32; 3] = shrink_to::<3, _, _>(array);
/// assert_eq!(array, [1, 2, 3]);
/// ```
#[must_use]
#[inline(always)]
pub fn shrink_to<const NEW_LEN: usize, T, const OLD_LEN: usize>(
    mut array: [T; OLD_LEN],
) -> [T; NEW_LEN] {
    const_assert_smaller!(NEW_LEN, OLD_LEN);

    let mut data = MaybeUninit::<[T; NEW_LEN]>::uninit();
    unsafe {
        ptr::copy_nonoverlapping(array.as_ptr(), data.as_mut_ptr().cast(), NEW_LEN);

        // Drop trailing items from the old array
        {
            let slice = &mut array[NEW_LEN..OLD_LEN];
            ptr::drop_in_place(slice);
        }

        mem::forget(array);
        MaybeUninit::assume_init(data)
    }
}

/// Shrinks an array, returning a new array with `NEW_LEN` elements.
///
/// # Examples
///
/// ```
/// # use vectral::utils::shrink_to_copy;
/// let array = [1, 2, 3, 4, 5];
/// let array: [i32; 2] = shrink_to_copy::<2, _, _>(array);
/// assert_eq!(array, [1i32, 2]);
/// ```
#[must_use]
#[inline(always)]
pub const fn shrink_to_copy<const NEW_LEN: usize, T: Copy, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
) -> [T; NEW_LEN] {
    const_assert_smaller_or_equal!(NEW_LEN, OLD_LEN);
    unsafe { shrink_to_copy_impl(array) }
}

#[must_use]
#[inline(always)]
const unsafe fn shrink_to_copy_impl<const NEW_LEN: usize, T: Copy, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
) -> [T; NEW_LEN] {
    let mut data = MaybeUninit::<[T; NEW_LEN]>::uninit();
    unsafe {
        ptr::copy_nonoverlapping(array.as_ptr(), data.as_mut_ptr().cast(), NEW_LEN);
        MaybeUninit::assume_init(data)
    }
}

/// Expands the array, inserting `NEW_LEN` - `OLD_LEN` instances of the value `to_append` at
/// the end of the array.
///
/// # Notes
///
/// A static assertion is used to ensure that `NEW_LEN` is always bigger than or equal to `OLD_LEN`.
///
/// # Examples
///
/// ```
/// # use vectral::utils::expand_to_copy;
///
/// let data = [1, 2, 3, 4];
/// let expanded = expand_to_copy::<6, _, _>(data, 0);
///
/// assert_eq!(expanded, [1, 2, 3, 4, 0, 0]);
/// ```
#[must_use]
#[inline(always)]
pub const fn expand_to_copy<const NEW_LEN: usize, T: Copy, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
    to_append: T,
) -> [T; NEW_LEN] {
    const_assert_larger_or_equal!(NEW_LEN, OLD_LEN);
    unsafe { expand_to_copy_impl(array, to_append) }
}

#[inline(always)]
const unsafe fn expand_to_copy_impl<const NEW_LEN: usize, T: Copy, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
    to_append: T,
) -> [T; NEW_LEN] {
    let mut data = [const { MaybeUninit::uninit() }; NEW_LEN];

    unsafe {
        let (left, right) = data.as_mut_slice().split_at_mut(OLD_LEN);

        ptr::copy_nonoverlapping(
            array.as_ptr().cast::<T>(),
            left.as_mut_ptr().cast(),
            OLD_LEN,
        );

        fill_copy(right, MaybeUninit::new(to_append));
        array_assume_init(data)
    }
}

/// Expands the array, inserting `NEW_LEN` - `OLD_LEN` clones of the value `to_append` at
/// the end of the array.
///
/// # Notes
///
/// A static assertion is used to ensure that `NEW_LEN` is always bigger than or equal to `OLD_LEN`.
///
/// # Examples
///
/// ```
/// # use vectral::utils::expand_to;
///
/// let data = ["a", "b", "c", "d"].map(String::from);
/// let expanded = expand_to::<6, _, _>(data, "f".to_string());
///
/// for (elem, expected) in expanded.iter().zip(&["a", "b", "c", "d", "f", "f"]) {
///     assert_eq!(&elem, expected);
/// }
/// ```
#[must_use]
#[inline(always)]
pub fn expand_to<const NEW_LEN: usize, T: Clone, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
    to_append: T,
) -> [T; NEW_LEN] {
    const_assert_larger_or_equal!(NEW_LEN, OLD_LEN);

    let mut data = const { MaybeUninit::<[T; NEW_LEN]>::uninit() };

    unsafe {
        let (left, right) = {
            let lp = data.as_mut_ptr().cast::<MaybeUninit<T>>();
            let rp = lp.add(OLD_LEN);
            (
                slice::from_raw_parts_mut(lp, OLD_LEN),
                slice::from_raw_parts_mut(rp, NEW_LEN - OLD_LEN),
            )
        };

        for (slot, data) in left.iter_mut().zip(array) {
            slot.write(data);
        }

        let mut iter = right.iter_mut();
        for slot in iter.by_ref().take((NEW_LEN - OLD_LEN).saturating_sub(1)) {
            slot.write(to_append.clone());
        }

        if let Some(slot) = iter.next() {
            slot.write(to_append);
        }

        MaybeUninit::assume_init(data)
    }
}

/// Returns a new array of size `NEW_LEN`.
///
/// If `NEW_LEN` is greater than `OLD_LEN`, the array will be extended by copying `to_append`.
/// If `NEW_LEN` is less than `OLD_LEN`, the array will be truncated.
///
/// # Examples
///
/// ```
/// use vectral::utils::arrays::resize_copy;
///
/// let my_data = [1, 2, 3];
///
/// let shrunk = resize_copy::<1, _, 3>(my_data.clone(), 240);
/// assert_eq!(shrunk.len(), 1);
/// assert_eq!(shrunk[0], 1);
///
/// let extended = resize_copy::<5, _, _>(my_data.clone(), 240);
/// assert_eq!(extended.len(), 5);
/// assert_eq!(extended[3], 240);
/// assert_eq!(extended[4], 240);
/// ```
#[must_use]
#[inline]
pub const fn resize_copy<const NEW_LEN: usize, T: Copy, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
    to_append: T,
) -> [T; NEW_LEN] {
    if NEW_LEN >= OLD_LEN {
        unsafe { expand_to_copy_impl(array, to_append) }
    } else {
        unsafe { shrink_to_copy_impl(array) }
    }
}

/// Returns a new array of size `NEW_LEN`.
///
/// If `NEW_LEN` is greater than `OLD_LEN`, the array will be extended by cloning `to_append`.
/// If `NEW_LEN` is less than `OLD_LEN`, the array will be truncated.
///
/// # Examples
///
/// ```
/// use vectral::utils::arrays::resize;
///
/// let my_data = [vec![1], vec![2], vec![3]];
///
/// let shrunk = resize::<1, _, 3>(my_data.clone(), vec![240]);
/// assert_eq!(shrunk.len(), 1);
/// assert_eq!(shrunk[0].as_slice(), &[1]);
///
/// let extended = resize::<5, _, _>(my_data.clone(), vec![240]);
/// assert_eq!(extended.len(), 5);
/// assert_eq!(&extended[3], &[240]);
/// assert_eq!(&extended[4], &[240]);
/// ```
#[must_use]
#[inline]
pub fn resize<const NEW_LEN: usize, T: Clone, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
    to_append: T,
) -> [T; NEW_LEN] {
    fn shrink_to_wrapper<const NEW_LEN: usize, T, const OLD_LEN: usize>(
        array: [T; OLD_LEN],
        _: T,
    ) -> [T; NEW_LEN] {
        shrink_to::<NEW_LEN, T, OLD_LEN>(array)
    }

    struct ChooseFunction<const NEW_LEN: usize, T, const OLD_LEN: usize>(PhantomData<T>);

    impl<const NEW_LEN: usize, const OLD_LEN: usize, T: Clone> ChooseFunction<NEW_LEN, T, OLD_LEN> {
        const FUNCTION: fn([T; OLD_LEN], T) -> [T; NEW_LEN] = if NEW_LEN >= OLD_LEN {
            expand_to::<NEW_LEN, T, OLD_LEN>
        } else {
            shrink_to_wrapper::<NEW_LEN, T, OLD_LEN>
        };
    }

    (ChooseFunction::<NEW_LEN, T, OLD_LEN>::FUNCTION)(array, to_append)
}

/// Flattens a nested array into a single flat array containing all elements.
///
/// # Examples
///
/// ```
/// # use vectral::utils::flatten;
/// let matrix = [[1, 2], [3, 4]];
/// let flat_matrix: [i32; _] = flatten(matrix);
/// assert_eq!(flat_matrix, [1, 2, 3, 4]);
/// ```
#[cfg(feature = "nightly")]
#[must_use]
#[inline(always)]
pub const fn flatten<T, const N0: usize, const N1: usize>(array: [[T; N0]; N1]) -> [T; N1 * N0] {
    let array = ManuallyDrop::new(array);
    unsafe { mem::transmute_copy(&array) }
}

#[must_use]
#[inline(always)]
pub const fn copied<T: Copy, const N: usize>(array: [&'_ T; N]) -> [T; N] {
    let mut result = [const { MaybeUninit::uninit() }; _];

    let mut i = 0;
    while i < N {
        unsafe {
            result
                .as_mut_ptr()
                .cast::<T>()
                .add(i)
                .write(**array.as_ptr().add(i));
        }
        i += 1;
    }

    unsafe { array_assume_init(result) }
}

/// Fills the given `slice` with copies of `element`.
///
/// # Examples
///
/// ```
/// use vectral::utils::arrays::fill_copy;
///
/// let mut arr = [0u8; 255];
/// fill_copy(&mut arr, 123);
///
/// for item in &arr {
///     assert_eq!(*item, 123);
/// }
/// ```
#[inline]
pub const fn fill_copy<T: Copy>(slice: &mut [T], element: T) {
    let mut i = 0;
    while i < slice.len() {
        unsafe {
            *array_get_unchecked_mut(slice, i) = element;
        }
        i += 1;
    }
}

/// Concatenates two arrays together, returning a new array containing all elements of `arr_0` followed
/// by all elements of `arr_1`.
///
/// # Examples
///
/// ```
/// use vectral::utils::arrays::concat;
///
/// let array_1 = [1.0, 3.0, 5.0];
/// let array_2 = [7.0, 9.0];
///
/// let joined = concat(array_1, array_2);
///
/// assert_eq!(&joined, &[1.0, 3.0, 5.0, 7.0, 9.0]);
/// ```
#[cfg(feature = "nightly")]
#[must_use]
#[inline(always)]
pub const fn concat<T, const N0: usize, const N1: usize>(
    arr_0: [T; N0],
    arr_1: [T; N1],
) -> [T; N0 + N1] {
    let mut result = [const { MaybeUninit::uninit() }; { N0 + N1 }];

    unsafe {
        ptr::copy_nonoverlapping(arr_0.as_ptr(), result.as_mut_ptr().cast(), N0);
        ptr::copy_nonoverlapping(arr_1.as_ptr(), result.as_mut_ptr().cast::<T>().add(N0), N1);
    }

    let _arrs = ManuallyDrop::new((arr_0, arr_1));
    unsafe { array_assume_init(result) }
}

/// Split the array at the `IDX` into two arrays.
///
/// # Examples
///
/// ```
/// # use vectral::utils::split;
/// let data = [1, 2, 3, 4, 5, 6];
/// let (first, second): ([_; 4], [_; _]) = split::<4, _, _>(data);
///
/// assert_eq!(&first, &[1, 2, 3, 4]);
/// assert_eq!(&second, &[5, 6]);
/// ```
#[cfg(feature = "nightly")]
#[must_use]
#[inline(always)]
pub const fn split<const IDX: usize, T, const SIZE: usize>(
    array: [T; SIZE],
) -> ([T; IDX], [T; SIZE - IDX]) {
    const_assert_smaller_or_equal!(IDX, SIZE);

    let (mut lhs, mut rhs) = (
        [const { MaybeUninit::uninit() }; _],
        [const { MaybeUninit::uninit() }; _],
    );

    unsafe {
        let (lhs_src, rhs_src) = array.as_slice().split_at_unchecked(IDX);
        ptr::copy_nonoverlapping(lhs_src.as_ptr(), lhs.as_mut_ptr().cast(), lhs_src.len());
        ptr::copy_nonoverlapping(rhs_src.as_ptr(), rhs.as_mut_ptr().cast(), rhs_src.len());
    }

    mem::forget(array);
    unsafe { (array_assume_init(lhs), array_assume_init(rhs)) }
}

#[must_use]
#[inline]
pub(crate) const fn array_get_checked<T>(array: &[T], index: usize) -> Option<&T> {
    if index < array.len() {
        unsafe { Some(array_get_unchecked(array, index)) }
    } else {
        None
    }
}

#[must_use]
#[inline]
pub(crate) const fn array_get_mut_checked<T>(array: &mut [T], index: usize) -> Option<&mut T> {
    if index < array.len() {
        unsafe { Some(array_get_unchecked_mut(array, index)) }
    } else {
        None
    }
}

#[must_use]
#[inline]
pub(crate) const unsafe fn array_get_unchecked<T>(array: &[T], index: usize) -> &T {
    unsafe { &*array.as_ptr().add(index) }
}

#[must_use]
#[inline]
pub(crate) const unsafe fn array_get_unchecked_mut<T>(array: &mut [T], index: usize) -> &mut T {
    unsafe { &mut *array.as_mut_ptr().add(index) }
}

#[must_use]
#[inline]
pub(crate) const unsafe fn array_assume_init<T, const N: usize>(
    array: [MaybeUninit<T>; N],
) -> [T; N] {
    unsafe { mem::transmute_copy(&array) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        array,
        sync::atomic::{AtomicUsize, Ordering},
    };

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
    fn test_shrink() {
        static NUM_INSTANCES: AtomicUsize = AtomicUsize::new(0usize);

        struct CountInstances;

        impl CountInstances {
            fn new() -> Self {
                NUM_INSTANCES.fetch_add(1, Ordering::SeqCst);
                Self
            }
        }

        impl Drop for CountInstances {
            fn drop(&mut self) {
                NUM_INSTANCES.fetch_sub(1, Ordering::SeqCst);
            }
        }

        let instances: [_; 5] = array::from_fn(|_| CountInstances::new());
        assert_eq!(NUM_INSTANCES.load(Ordering::SeqCst), 5);

        let _instances = shrink_to::<4, _, _>(instances);
        assert_eq!(NUM_INSTANCES.load(Ordering::SeqCst), 4);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_join() {
        const FST: [i32; 5] = [1, 2, 3, 4, 5];
        const SND: [i32; 5] = [6, 7, 8, 9, 10];

        const JOINED: [i32; 10] = concat(FST, SND);
        assert_eq!(JOINED, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        const SPLIT_FIRST_EMPTY: ([i32; 0], [i32; 10]) = split::<0, _, _>(JOINED);
        const SPLIT_SECOND_EMPTY: ([i32; 10], [i32; 0]) = split::<10, _, _>(JOINED);

        assert_eq!(SPLIT_FIRST_EMPTY.0, SPLIT_SECOND_EMPTY.1);
        assert_eq!(SPLIT_FIRST_EMPTY.1, SPLIT_SECOND_EMPTY.0);

        const SPLIT: ([i32; 5], [i32; 5]) = split(JOINED);
        assert_eq!(FST, SPLIT.0);
        assert_eq!(SND, SPLIT.1);
    }
}
