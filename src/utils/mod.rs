use crate::{utils::num::{ClosedAdd, ClosedMul, One, Zero}, const_assert_smaller, const_assert_larger};
use core::{
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Add, Mul},
    ptr,
};

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
}

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

/// Zips two arrays together and applies the function `f` to each memberwise element, returning a fixed
/// size array of the results.
///
/// It is preferred to use this function over chaining the [`zip()`] and [`map()`] methods together (e.g.
/// `zip(array).map(|(x, y)| x + y);`, as it avoids allocating an intermediate array to store the zipped
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
    let (lhs, rhs) = (ManuallyDrop::new(lhs), ManuallyDrop::new(rhs));
    let mut result = [const { MaybeUninit::<Res>::uninit() }; N];

    for i in 0..N {
        unsafe {
            let slot = result.get_unchecked_mut(i);
            let lhs = ptr::read(lhs.get_unchecked(i));
            let rhs = ptr::read(rhs.get_unchecked(i));

            slot.write(f(lhs, rhs));
        }
    }

    unsafe { MaybeUninit::array_assume_init(result) }
}

/// Zips two fixed-size arrays together, returning a fixed size array of tuples.
///
/// If the result of this function will be immediately used as an intermediate calculation,
/// it is preferred to use the [`zip_map()`] function.
///
/// # Examples
///
/// ```
/// # use vectral::utils::zip;
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
#[must_use]
#[inline(always)]
pub const fn zip<T, U, const N: usize>(lhs: [T; N], rhs: [U; N]) -> [(T, U); N] {
    let mut zipped = [const { MaybeUninit::<(T, U)>::uninit() }; N];

    let mut i = 0;
    while i < N {
        unsafe {
            let slot = zipped.get_unchecked_mut(i);
            let lhs = ptr::read(lhs.get_unchecked(i));
            let rhs = ptr::read(rhs.get_unchecked(i));

            slot.write((lhs, rhs));
        }
        i += 1;
    }

    let _arrays = (ManuallyDrop::new(lhs), ManuallyDrop::new(rhs));

    unsafe { MaybeUninit::array_assume_init(zipped) }
}

#[must_use]
#[inline(always)]
pub const fn unzip<T, U, const N: usize>(array: [(T, U); N]) -> ([T; N], [U; N]) {
    let (mut lhs, mut rhs) = (
        [const { MaybeUninit::<T>::uninit() }; N], 
        [const { MaybeUninit::<U>::uninit() }; N],
    );

    let mut i = 0;
    while i < N {
        unsafe {
            let slot = array.get_unchecked(i);
            lhs.get_unchecked_mut(i).write(ptr::read(&slot.0));
            rhs.get_unchecked_mut(i).write(ptr::read(&slot.1));
        }

        i += 1;
    }

    mem::forget(array);

    unsafe {
        let lhs = MaybeUninit::array_assume_init(lhs);
        let rhs = MaybeUninit::array_assume_init(rhs);
        (lhs, rhs)
    }
}

/// Shrinks a fixed size array, removing the final element.
///
/// Examples
///
/// ```
/// # use vectral::utils::shrink;
/// let array = [1, 2, 3, 4, 5];
/// let array = shrink(array);
/// assert_eq!(array, [1, 2, 3, 4]);
/// ```
#[must_use]
#[inline(always)]
pub fn shrink<T, const N: usize>(array: [T; N]) -> [T; N - 1] {
    shrink_to(array)
}

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

        {
            let slice = &mut array[NEW_LEN..OLD_LEN];
            ptr::drop_in_place(slice);
        }

        let _array = ManuallyDrop::new(array);
        MaybeUninit::assume_init(data)
    }
}

#[must_use]
#[inline(always)]
pub const fn shrink_to_copy<const NEW_LEN: usize, T: Copy, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
) -> [T; NEW_LEN] {
    const_assert_smaller!(NEW_LEN, OLD_LEN);

    let mut data = MaybeUninit::<[T; NEW_LEN]>::uninit();
    unsafe {
        ptr::copy_nonoverlapping(array.as_ptr(), data.as_mut_ptr().cast(), NEW_LEN);
        MaybeUninit::assume_init(data)
    }
}

/// Expands the array, inserting the value `to_append` into the end of the array.
///
/// # Examples
///
/// ```
/// # use vectral::utils::expand;
/// let array = [1, 2, 3];
/// let expanded = expand(array, 4);
/// assert_eq!(expanded, [1, 2, 3, 4]);
/// ```
#[must_use]
#[inline(always)]
pub const fn expand<T, const N: usize>(array: [T; N], to_append: T) -> [T; N + 1] {
    concat(array, [to_append])
}

/// Expands the array, inserting `NEW_LEN` - `OLD_LEN` instances of the value `to_append` at the end of the array.
///
/// # Notes
///
/// A static assertion is used to ensure that `NEW_LEN` is always bigger than `OLD_LEN`.
#[must_use]
#[inline(always)]
pub const fn expand_to<const NEW_LEN: usize, T: Copy, const OLD_LEN: usize>(
    array: [T; OLD_LEN],
    to_append: T,
) -> [T; NEW_LEN] {
    const_assert_larger!(NEW_LEN, OLD_LEN);

    let mut data = MaybeUninit::<[T; NEW_LEN]>::uninit();

    unsafe {
        ptr::copy_nonoverlapping(
            array.as_ptr().cast::<T>(),
            data.as_mut_ptr().cast(),
            OLD_LEN,
        );
        let mut i = OLD_LEN;
        while i < NEW_LEN {
            ptr::write(
                data.as_mut_ptr().cast::<T>().add(i),
                mem::transmute_copy(&to_append),
            );
            i += 1;
        }

        let _ = ManuallyDrop::new((to_append, array));
        data.assume_init()
    }
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
#[must_use]
#[inline(always)]
pub const fn flatten<T, const N0: usize, const N1: usize>(array: [[T; N0]; N1]) -> [T; N1 * N0] {
    let array = ManuallyDrop::new(array);
    unsafe { mem::transmute_copy(&array) }
}

#[must_use]
#[inline(always)]
pub fn cloned<T: Clone, const N: usize>(array: [&'_ T; N]) -> [T; N] {
    array.map(Clone::clone)
}

#[must_use]
#[inline(always)]
pub const fn copied<T: Copy, const N: usize>(array: [&'_ T; N]) -> [T; N] {
    let mut result = [const { MaybeUninit::uninit() }; _];

    let mut i = 0;
    while i < N {
        unsafe {
            result
                .get_unchecked_mut(i)
                .write(**array.get_unchecked(i));
        }
        i += 1;
    }

    unsafe { MaybeUninit::array_assume_init(result) }
}

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
    unsafe { MaybeUninit::array_assume_init(result) }
}

#[must_use]
#[inline(always)]
pub const fn split<const IDX: usize, T, const SIZE: usize>(
    array: [T; SIZE],
) -> ([T; IDX], [T; SIZE - IDX]) {
    const_assert_smaller!(IDX, SIZE);

    let (mut lhs, mut rhs) = (
        [const { MaybeUninit::uninit() }; _],
        [const { MaybeUninit::uninit() }; _],
    );

    unsafe {
        ptr::copy_nonoverlapping(array.as_ptr(), lhs.as_mut_ptr().cast::<T>(), IDX);
        ptr::copy_nonoverlapping(
            array.as_ptr().add(IDX),
            rhs.as_mut_ptr().cast::<T>(),
            SIZE - IDX,
        );

        let _ = ManuallyDrop::new(array);

        (
            MaybeUninit::array_assume_init(lhs),
            MaybeUninit::array_assume_init(rhs),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{
        array,
        sync::atomic::{AtomicUsize, Ordering},
    };

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

        let _instances = shrink(instances);
        assert_eq!(NUM_INSTANCES.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_join() {
        const FST: [i32; 5] = [1, 2, 3, 4, 5];
        const SND: [i32; 5] = [6, 7, 8, 9, 10];

        const JOINED: [i32; 10] = concat(FST, SND);
        assert_eq!(JOINED, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        const SPLIT: ([i32; 5], [i32; 5]) = split(JOINED);
        assert_eq!(FST, SPLIT.0);
        assert_eq!(SND, SPLIT.1);
    }
}
