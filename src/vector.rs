// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    const_assert_larger,
    point::Point,
    utils::{
        array_get_checked, array_get_mut_checked, array_get_unchecked, array_get_unchecked_mut,
        expand_to,
        num::{
            ClosedAdd, ClosedMul, ClosedNeg, ClosedSub, One, Sqrt, Zero,
            checked::{CheckedDiv, CheckedMul},
        },
        reversed, shrink_to, sum, zip_map,
    },
};
#[cfg(feature = "nightly")]
use crate::{
    matrix::{Matrix, TransformHomogeneous},
    transform::{Transform, Translate},
    utils::{concat, expand, shrink, split},
};
#[cfg(feature = "simd")]
use crate::{
    simd::{SimdAdd, SimdDiv, SimdMul, SimdSub, SimdValue},
    utils::num::ClosedDiv,
};
#[cfg(feature = "serde")]
use core::marker::PhantomData;
#[cfg(feature = "simd")]
use core::simd::{Simd, SimdElement};
use core::{
    array::{self, IntoIter},
    borrow::{Borrow, BorrowMut},
    fmt,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    ptr,
    slice::{self, Iter, IterMut},
};
#[cfg(feature = "serde")]
use serde_core::{
    de::{self, Deserialize, Deserializer, Error, Expected, SeqAccess},
    ser::{Serialize, SerializeTupleStruct, Serializer},
};

#[repr(C)]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Vector<T = f32, const N: usize = 3> {
    data: [T; N],
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for Vector<T, N> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.write_str("Vector ")?;
        let mut debug_list = fmtr.debug_list();
        debug_list.entries(self.data.iter());
        debug_list.finish()
    }
}

impl<T: Default, const N: usize> Default for Vector<T, N> {
    #[inline]
    fn default() -> Self {
        Self {
            data: array::from_fn(|_| Default::default()),
        }
    }
}

pub type Vector0<T = f32> = Vector<T, 0>;
pub type Vector1<T = f32> = Vector<T, 1>;
pub type Vector2<T = f32> = Vector<T, 2>;
pub type Vector3<T = f32> = Vector<T, 3>;
pub type Vector4<T = f32> = Vector<T, 4>;

impl_coerce_to_fields! {
    Vector<{T, 1}> => X,
    Vector<{T, 2}> => Xy,
    Vector<{T, 3}> => Xyz,
    Vector<{T, 4}> => Xyzw,
}

impl<T: One + Zero, const N: usize> Vector<T, N> {
    /// A constant representing a vector where the X-axis is set to `1`, and other
    /// elements are set to `0`.
    ///
    /// This constant is not defined when the length of the vector is less than 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector3;
    /// let vector = Vector3::<i32>::X;
    /// assert_eq!(vector.x, 1);
    /// assert_eq!(vector.y, 0);
    /// assert_eq!(vector.z, 0);
    /// ```
    pub const X: Self = Vector::unit_n::<0>();

    /// A constant representing a vector where the Y-axis is set to `1`, and other
    /// elements are set to `0`.
    ///
    /// This constant is not defined when the length of the vector is less than 2.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector3;
    /// let vector = Vector3::<i32>::Y;
    /// assert_eq!(vector.x, 0);
    /// assert_eq!(vector.y, 1);
    /// assert_eq!(vector.z, 0);
    /// ```
    pub const Y: Self = Vector::unit_n::<1>();

    /// A constant representing a vector where the Z-axis is set to `1`, and other
    /// elements are set to `0`.
    ///
    /// This constant is not defined when the length of the vector is less than 3.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector3;
    /// let vector = Vector3::<i32>::Z;
    /// assert_eq!(vector.x, 0);
    /// assert_eq!(vector.y, 0);
    /// assert_eq!(vector.z, 1);
    /// ```
    pub const Z: Self = Vector::unit_n::<2>();

    /// A constant representing a vector where the W-axis is set to `1`, and other
    /// elements are set to `0`.
    ///
    /// This constant is not defined when the length of the vector is less than 4.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector4;
    /// let vector = Vector4::<i32>::W;
    /// assert_eq!(vector.x, 0);
    /// assert_eq!(vector.y, 0);
    /// assert_eq!(vector.z, 0);
    /// assert_eq!(vector.w, 1);
    /// ```
    pub const W: Self = Vector::unit_n::<3>();

    /// Returns a new vector facing towards the given dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector;
    /// let vector = Vector::<i32, 6>::unit_n::<2>();
    /// assert_eq!(vector.to_array(), [0, 0, 1, 0, 0, 0]);
    /// ```
    ///
    /// This method will fail to compile if the given `DIM` is greater
    /// than the length of the vector.
    ///
    /// ```compile_fail
    /// use vectral::vector::Vector;
    ///
    /// let _ = Vector::<i32, 6>::unit_n::<12>();
    /// ```
    #[must_use]
    #[inline]
    pub const fn unit_n<const DIM: usize>() -> Self {
        const_assert_larger!(N, DIM);

        let mut vector = Vector::new([const { MaybeUninit::new(T::ZERO) }; N]);
        unsafe {
            vector.get_unchecked_mut(DIM).write(T::ONE);
            Vector::assume_init(vector)
        }
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub const LENGTH: usize = N;

    /// Returns a new vector, initializing every element from the given `array`.
    ///
    /// # Example
    ///
    /// ```
    /// # use vectral::vector::Vector;
    /// let vector: Vector<i32, 5> = Vector::new([1, 2, 3, 4, 5]);
    /// # let _ = vector;
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(array: [T; N]) -> Self {
        Self { data: array }
    }

    #[must_use]
    #[inline]
    pub fn from_fn<F: FnMut(usize) -> T>(f: F) -> Self {
        Self::new(array::from_fn(f))
    }

    #[must_use]
    #[inline]
    pub const fn uninit() -> Vector<MaybeUninit<T>, N> {
        Vector {
            data: [const { MaybeUninit::uninit() }; N],
        }
    }

    #[inline]
    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> Vector<U, N> {
        Vector {
            data: self.data.map(f),
        }
    }

    #[inline]
    pub fn zip_map<U, Ret, F: FnMut(T, U) -> Ret>(self, rhs: Vector<U, N>, f: F) -> Vector<Ret, N> {
        Vector {
            data: zip_map(self.data, rhs.data, f),
        }
    }

    #[must_use]
    #[inline]
    pub fn dot<U>(self, rhs: Vector<U, N>) -> T::Output
    where
        T: Mul<U>,
        T::Output: Zero + ClosedAdd,
    {
        sum(self.into_iter().zip(rhs).map(|(x, y)| x * y))
    }

    #[must_use]
    #[inline]
    pub fn elementwise_mul<U>(self, rhs: Vector<U, N>) -> Vector<T::Output, N>
    where
        T: Mul<U>,
    {
        self.zip_map(rhs, Mul::mul)
    }

    #[must_use]
    #[inline]
    pub fn elementwise_div<U>(self, rhs: Vector<U, N>) -> Vector<T::Output, N>
    where
        T: Div<U>,
    {
        self.zip_map(rhs, Div::div)
    }

    #[must_use]
    #[inline]
    pub fn elementwise_add<U>(self, rhs: Vector<U, N>) -> Vector<T::Output, N>
    where
        T: Add<U>,
    {
        self.zip_map(rhs, Add::add)
    }

    #[must_use]
    #[inline]
    pub fn elementwise_sub<U>(self, rhs: Vector<U, N>) -> Vector<T::Output, N>
    where
        T: Sub<U>,
    {
        self.zip_map(rhs, Sub::sub)
    }

    #[must_use]
    #[inline]
    pub const fn get(&self, index: usize) -> Option<&T> {
        array_get_checked(&self.data, index)
    }

    #[must_use]
    #[inline]
    pub const fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        array_get_mut_checked(&mut self.data, index)
    }

    #[must_use]
    #[inline]
    pub const unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { array_get_unchecked(&self.data, index) }
    }

    #[must_use]
    #[inline]
    pub const unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe { array_get_unchecked_mut(&mut self.data, index) }
    }

    #[must_use]
    #[inline]
    pub const fn each_ref(&self) -> Vector<&T, N> {
        Vector::new(self.data.each_ref())
    }

    #[must_use]
    #[inline]
    pub const fn each_mut(&mut self) -> Vector<&mut T, N> {
        Vector::new(self.data.each_mut())
    }

    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[must_use]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), N) }
    }

    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), N) }
    }

    #[deprecated(note = "use Vector::to_array instead")]
    #[must_use]
    #[inline]
    pub const fn into_array(self) -> [T; N] {
        self.to_array()
    }

    #[must_use]
    #[inline]
    pub const fn to_array(self) -> [T; N] {
        let array = unsafe { ptr::read(&self.data) };
        let _this = ManuallyDrop::new(self);
        array
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub fn shrink(self) -> Vector<T, { N - 1 }> {
        Vector {
            data: shrink(self.data),
        }
    }

    #[must_use]
    #[inline]
    pub fn shrink_to<const N_NEW: usize>(self) -> Vector<T, N_NEW> {
        Vector {
            data: shrink_to(self.data),
        }
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn expand(self, to_append: T) -> Vector<T, { N + 1 }> {
        let data = unsafe { ptr::read(&self.data) };
        let _self = ManuallyDrop::new(self);
        Vector {
            data: expand(data, to_append),
        }
    }

    #[must_use]
    #[inline]
    pub const fn into_point(self) -> Point<T, N> {
        let point = unsafe { Point::new(ptr::read(&self.data)) };
        let _self = ManuallyDrop::new(self);
        point
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Reverses the elements in the `Vector`.
    ///
    /// # Example
    ///
    /// ```
    /// use vectral::vector::Vector;
    ///
    /// let vec = Vector::new([1, 2, 3, 4, 5]);
    /// let rev = vec.reversed();
    ///
    /// assert_eq!(rev.to_array(), [5, 4, 3, 2, 1]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn reversed(self) -> Vector<T, N> {
        let reversed = unsafe { reversed(ptr::read(&self.data)) };
        mem::forget(self);

        Vector { data: reversed }
    }

    #[inline]
    pub const fn reverse(&mut self) {
        self.data.reverse();
    }

    /// Concatenates two vectors together to form a larger vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector;
    /// let v1 = Vector::new([1, 2, 3]);
    /// let v2 = Vector::new([4, 5, 6]);
    /// let v3 = v1.concat(v2);
    ///
    /// assert_eq!(v3.to_array(), [1, 2, 3, 4, 5, 6]);
    /// ```
    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn concat<const N1: usize>(self, other: Vector<T, N1>) -> Vector<T, { N + N1 }> {
        let data = {
            let data = unsafe { concat(ptr::read(&self.data), ptr::read(&other.data)) };
            let _ = ManuallyDrop::new((self, other));
            data
        };
        Vector::new(data)
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub fn split<const IDX: usize>(self) -> (Vector<T, IDX>, Vector<T, { N - IDX }>) {
        let (l, r) = split(self.data);
        (Vector::new(l), Vector::new(r))
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdMul for Vector<T, N>
where
    Simd<T, N>: ClosedMul,
{
    type Output = Self;
    #[inline]
    fn simd_mul(self, rhs: Self) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::from_array(rhs.to_array());
        let res = lhs * rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdMul<T> for Vector<T, N>
where
    Simd<T, N>: ClosedMul,
{
    type Output = Self;
    #[inline]
    fn simd_mul(self, rhs: T) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::splat(rhs);
        let res = lhs * rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdDiv for Vector<T, N>
where
    Simd<T, N>: ClosedDiv,
{
    type Output = Self;
    #[inline]
    fn simd_div(self, rhs: Self) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::from_array(rhs.to_array());
        let res = lhs / rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdDiv<T> for Vector<T, N>
where
    Simd<T, N>: ClosedDiv,
{
    type Output = Self;
    #[inline]
    fn simd_div(self, rhs: T) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::splat(rhs);
        let res = lhs / rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdAdd for Vector<T, N>
where
    Simd<T, N>: ClosedAdd,
{
    type Output = Self;
    #[inline]
    fn simd_add(self, rhs: Self) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::from_array(rhs.to_array());
        let res = lhs + rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdAdd<T> for Vector<T, N>
where
    Simd<T, N>: ClosedAdd,
{
    type Output = Self;
    #[inline]
    fn simd_add(self, rhs: T) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::splat(rhs);
        let res = lhs + rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdSub for Vector<T, N>
where
    Simd<T, N>: ClosedSub,
{
    type Output = Self;
    #[inline]
    fn simd_sub(self, rhs: Self) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::from_array(rhs.to_array());
        let res = lhs - rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> SimdSub<T> for Vector<T, N>
where
    Simd<T, N>: ClosedSub,
{
    type Output = Self;
    #[inline]
    fn simd_sub(self, rhs: T) -> Self::Output {
        let lhs = Simd::from_array(self.to_array());
        let rhs = Simd::splat(rhs);
        let res = lhs - rhs;
        Self::new(res.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement + ClosedAdd + Zero, const N: usize> Vector<T, N>
where
    Simd<T, N>: ClosedMul,
{
    #[must_use]
    #[inline]
    pub fn simd_dot<I: Into<SimdValue<Self>>>(self, rhs: I) -> T {
        let res = (SimdValue(self) * rhs.into()).to_array();
        sum(res.into_iter())
    }

    #[must_use]
    #[inline]
    pub fn simd_len_squared(self) -> T {
        Self::simd_dot(self, self)
    }

    #[must_use]
    #[inline]
    pub fn simd_len(self) -> T
    where
        T: Sqrt,
    {
        Self::simd_len_squared(self).sqrt()
    }
}

impl<T, const N: usize> Vector<MaybeUninit<T>, N> {
    #[must_use]
    #[inline]
    pub const unsafe fn assume_init(self) -> Vector<T, N> {
        Vector {
            data: unsafe { MaybeUninit::array_assume_init(self.data) },
        }
    }
}

impl<T: ClosedMul + Copy + Zero + ClosedAdd, const N: usize> Vector<T, N> {
    #[must_use]
    #[inline]
    pub fn len_squared(self) -> <T as Add>::Output {
        Self::dot(self, self)
    }
}

impl<T: ClosedMul + Copy + ClosedAdd + Zero + Sqrt, const N: usize> Vector<T, N> {
    #[must_use]
    #[inline]
    pub fn len(self) -> <T as Add>::Output {
        self.len_squared().sqrt()
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: ClosedMul + Copy + ClosedAdd + Div + Zero + Sqrt,
{
    #[must_use]
    #[inline]
    pub fn normalized_unchecked(self) -> Vector<<T as Div>::Output, N> {
        let len = self.len();
        self.div(len)
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: ClosedMul + Copy + ClosedAdd + CheckedDiv<Output = T> + Zero + Sqrt,
{
    #[must_use]
    #[inline]
    pub fn normalized(self) -> Vector<T, N> {
        self.normalized_checked().unwrap_or(Zero::ZERO)
    }

    #[must_use]
    #[inline]
    pub fn normalized_checked(self) -> Option<Vector<T, N>> {
        let len = self.len();
        self.checked_div(len)
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: ClosedMul + Copy + ClosedAdd + DivAssign<<T as Add>::Output> + Zero + Sqrt + PartialEq,
{
    #[inline]
    pub fn normalize(&mut self) {
        let len = self.len();
        if len != Zero::ZERO {
            self.div_assign(len);
        } else {
            *self = Zero::ZERO;
        }
    }
}

impl<T: Copy + ClosedMul + ClosedSub + ClosedNeg> Vector3<T> {
    #[must_use]
    #[inline]
    pub fn cross(self, rhs: Self) -> Vector3<T> {
        let [x0, y0, z0] = self.to_array();
        let [x1, y1, z1] = rhs.to_array();

        let x = (y0 * z1) - (z0 * y1);
        let y = (x0 * z1) - (z0 * x1);
        let z = (x0 * y1) - (y0 * x1);

        Vector3::new([x, y.neg(), z])
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement + ClosedNeg> Vector3<T>
where
    Simd<T, 3>: ClosedMul + ClosedSub,
{
    #[must_use]
    #[inline]
    pub fn simd_cross<I: Into<SimdValue<Self>>>(self, rhs: I) -> Self {
        let [x0, y0, z0] = self.to_array();
        let [x1, y1, z1] = rhs.into().to_array();

        let v0 = Simd::from_array([y0, x0, x0]);
        let v1 = Simd::from_array([z1, z1, y1]);

        let v2 = Simd::from_array([z0, z0, y0]);
        let v3 = Simd::from_array([y1, x1, x1]);

        let lhs = v0 * v1;
        let rhs = v2 * v3;
        let result = lhs - rhs;

        let [x, y, z] = result.to_array();
        Vector3::new([x, y.neg(), z])
    }
}

impl<T: Copy, const N: usize> Vector<T, N> {
    /// Create a new `Vector` with each element set to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector;
    /// let vector = Vector::<_, 12>::splat(21);
    /// vector.into_iter().for_each(|elem| assert_eq!(elem, 21));
    /// ```
    #[must_use]
    #[inline]
    pub const fn splat(value: T) -> Self {
        Vector { data: [value; N] }
    }

    /// Expands the `Vector` to the new given size, where each new value appended
    /// on the end is set to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector;
    /// let vector = Vector::new([1, 2]);
    /// let expanded = vector.expand_to::<4>(0);
    /// assert_eq!(expanded.to_array(), [1, 2, 0, 0]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn expand_to<const N1: usize>(self, value: T) -> Vector<T, N1> {
        Vector {
            data: expand_to(self.data, value),
        }
    }

    /// Swizzles the vector using the given `swizzle_vec`, returning a new vector where each
    /// element is the item at the index of `swizzle_vec`.
    ///
    /// # Notes
    ///
    /// If any index given in `swizzle_vec` is out of bounds, then this method will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector;
    /// let vector = Vector::new([1, 2, 8, 9, 3, 1, 4]);
    /// let swizzled = vector.try_swizzle(&[1, 4, 0, 0]).unwrap();
    /// assert_eq!(swizzled.to_array(), [2, 3, 1, 1]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn try_swizzle<const N1: usize>(
        self,
        swizzle_vec: &[usize; N1],
    ) -> Option<Vector<T, N1>> {
        let mut vec = Vector::<_, N1>::uninit();

        let mut i = 0;
        while i < N1 {
            unsafe {
                let swizzle_idx = *swizzle_vec.as_ptr().add(i);
                if swizzle_idx >= N {
                    return None;
                }

                let swizzle_elem = *self.get_unchecked(swizzle_idx);
                let slot = vec.get_unchecked_mut(i);
                slot.write(swizzle_elem);
            }

            i += 1;
        }

        unsafe { Some(Vector::assume_init(vec)) }
    }

    /// Swizzles the vector using the given `swizzle_vec`, returning a new vector where each
    /// element is the item at the index of `swizzle_vec`.
    ///
    /// # Notes
    ///
    /// If any index given in `swizzle_vec` is out of bounds, then the corresponding element from the
    /// `or` `Vector` will be used instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::vector::Vector;
    ///
    /// let vector = Vector::new([1, 2, 8, 9, 3, 1, 4]);
    /// let swizzled = vector.swizzle_or(&[1, 4, 0, 900], &Vector::new([6, 7, 8, 19]));
    ///
    /// assert_eq!(swizzled.to_array(), [2, 3, 1, 19]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn swizzle_or<const N1: usize>(
        self,
        swizzle_vec: &[usize; N1],
        or: &Vector<T, N1>,
    ) -> Vector<T, N1> {
        let mut vec = Vector::<_, N1>::uninit();

        let mut i = 0;
        while i < N1 {
            unsafe {
                let swizzle_idx = *swizzle_vec.as_ptr().add(i);

                let swizzle_elem = if swizzle_idx >= N {
                    *or.get_unchecked(i)
                } else {
                    *self.get_unchecked(swizzle_idx)
                };
                let slot = vec.get_unchecked_mut(i);
                slot.write(swizzle_elem);
            }

            i += 1;
        }

        unsafe { Vector::assume_init(vec) }
    }

    /// Swizzles the vector using the given `swizzle_vec`, returning a new vector where each
    /// element is the item at the index of `swizzle_vec`.
    ///
    /// # Panics
    ///
    /// If any index given in `swizzle_vec` is out of bounds, then this method will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::vector::Vector;
    /// let vector = Vector::new([1, 2, 8, 9, 3, 1, 4]);
    /// let swizzled = vector.swizzle(&[1, 4, 0, 0]);
    /// assert_eq!(swizzled.to_array(), [2, 3, 1, 1]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn swizzle<const N1: usize>(self, swizzle_vec: &[usize; N1]) -> Vector<T, N1> {
        match self.try_swizzle(swizzle_vec) {
            Some(vector) => vector,
            None => panic!("swizzle index out of bounds"),
        }
    }
}

impl<T: Zero, const N: usize> Zero for Vector<T, N> {
    const ZERO: Self = Self::new(Zero::ZERO);
}

#[cfg(feature = "nightly")]
impl<T, const N: usize> TransformHomogeneous<N> for Vector<T, N>
where
    T: Zero + One + PartialEq + Copy + DivAssign + ClosedMul + ClosedAdd,
{
    type Scalar = T;

    #[inline]
    fn transform_homogeneous(self, matrix: Matrix<Self::Scalar, { N + 1 }, { N + 1 }>) -> Self {
        let homogenous = self.expand_to(T::ONE);
        let mut transformed = matrix * homogenous;
        let w = transformed[N];
        if w != T::ONE {
            transformed.div_assign(w);
        }

        transformed.shrink_to()
    }
}

impl<T: Neg, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T::Output, N>;
    #[inline]
    fn neg(self) -> Self::Output {
        self.map(|elem| elem.neg())
    }
}

impl<T: Mul + Copy, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T::Output, N>;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.map(|elem| elem * rhs)
    }
}

impl<T: Copy + CheckedMul, const N: usize> CheckedMul<T> for Vector<T, N> {
    #[inline]
    fn checked_mul(self, rhs: T) -> Option<Self::Output> {
        let mut v: Vector<_, N> = Vector::uninit();

        for (i, elem) in self.into_iter().enumerate() {
            let res = elem.checked_mul(rhs)?;
            unsafe {
                v.get_unchecked_mut(i).write(res);
            }
        }

        unsafe { Some(Vector::assume_init(v)) }
    }
}

impl<T: MulAssign<U>, U: Copy, const N: usize> MulAssign<U> for Vector<T, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: U) {
        for elem in &mut self.data {
            elem.mul_assign(rhs);
        }
    }
}

impl<T: Div<U>, U: Copy, const N: usize> Div<U> for Vector<T, N> {
    type Output = Vector<T::Output, N>;
    #[inline]
    fn div(self, rhs: U) -> Self::Output {
        self.map(|elem| elem / rhs)
    }
}

impl<T: CheckedDiv<U>, U: Copy, const N: usize> CheckedDiv<U> for Vector<T, N> {
    #[inline]
    fn checked_div(self, rhs: U) -> Option<Self::Output> {
        let mut v: Vector<_, N> = Vector::uninit();

        for (i, elem) in self.into_iter().enumerate() {
            let res = elem.checked_div(rhs)?;
            unsafe {
                v.get_unchecked_mut(i).write(res);
            }
        }

        unsafe { Some(Vector::assume_init(v)) }
    }
}

impl<T: DivAssign<U>, U: Copy, const N: usize> DivAssign<U> for Vector<T, N> {
    #[inline]
    fn div_assign(&mut self, rhs: U) {
        for elem in self {
            elem.div_assign(rhs);
        }
    }
}

impl<T: Add<U>, U, const N: usize> Add<Vector<U, N>> for Vector<T, N> {
    type Output = Vector<T::Output, N>;
    #[inline]
    fn add(self, rhs: Vector<U, N>) -> Self::Output {
        self.zip_map(rhs, Add::add)
    }
}

impl<T: AddAssign<U>, U, const N: usize> AddAssign<Vector<U, N>> for Vector<T, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Vector<U, N>) {
        for (l, r) in self.data.iter_mut().zip(rhs) {
            l.add_assign(r);
        }
    }
}

impl<T: Sub<U>, U, const N: usize> Sub<Vector<U, N>> for Vector<T, N> {
    type Output = Vector<T::Output, N>;
    #[inline]
    fn sub(self, rhs: Vector<U, N>) -> Self::Output {
        self.zip_map(rhs, Sub::sub)
    }
}

impl<T: SubAssign<U>, U, const N: usize> SubAssign<Vector<U, N>> for Vector<T, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Vector<U, N>) {
        for (l, r) in self.data.iter_mut().zip(rhs) {
            l.sub_assign(r);
        }
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;
    #[track_caller]
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        Index::index(&self.data, index)
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.data, index)
    }
}

impl<T, const N: usize> AsRef<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.data
    }
}

impl<T, const N: usize> AsMut<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }
}

impl<T, const N: usize> Borrow<[T; N]> for Vector<T, N> {
    #[inline]
    fn borrow(&self) -> &[T; N] {
        &self.data
    }
}

impl<T, const N: usize> BorrowMut<[T; N]> for Vector<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }
}

impl<T, const N: usize> AsRef<[T]> for Vector<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data[..]
    }
}

impl<T, const N: usize> AsMut<[T]> for Vector<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

impl<T, const N: usize> Borrow<[T]> for Vector<T, N> {
    #[inline]
    fn borrow(&self) -> &[T] {
        &self.data[..]
    }
}

impl<T, const N: usize> BorrowMut<[T]> for Vector<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self.data[..]
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
    #[inline]
    fn from(value: Vector<T, N>) -> Self {
        value.data
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    #[inline]
    fn from(value: [T; N]) -> Self {
        Self { data: value }
    }
}

impl<T, const N: usize> From<Point<T, N>> for Vector<T, N> {
    #[inline]
    fn from(value: Point<T, N>) -> Self {
        Self {
            data: value.to_array(),
        }
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> From<Simd<T, N>> for Vector<T, N> {
    #[inline]
    fn from(value: Simd<T, N>) -> Self {
        Self::new(value.to_array())
    }
}

#[cfg(feature = "nightly")]
impl<T, const DIM: usize> Translate<DIM> for Vector<T, DIM>
where
    T: Zero + One + PartialEq + Copy + DivAssign + ClosedMul + ClosedAdd,
    Matrix<T, { DIM + 1 }, { DIM + 1 }>: Sized,
{
    type Scalar = T;
    #[inline]
    fn translated<Trans: Transform<DIM, Scalar = Self::Scalar>>(&self, transform: &Trans) -> Self {
        self.transform_homogeneous(transform.get_homogeneous().resize())
    }

    #[inline]
    fn translate_by<Trans: Transform<DIM, Scalar = Self::Scalar>>(&mut self, transform: &Trans) {
        *self = self.transform_homogeneous(transform.get_homogeneous().resize());
    }
}

impl<T, const N: usize> IntoIterator for Vector<T, N> {
    type IntoIter = IntoIter<T, N>;
    type Item = T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Vector<T, N> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Vector<T, N> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector2<T>> for Vector<T, 2> {
    #[inline]
    fn from(mint::Vector2 { x, y }: mint::Vector2<T>) -> Self {
        Self::new([x, y])
    }
}

#[cfg(feature = "mint")]
impl<T> From<Vector<T, 2>> for mint::Vector2<T> {
    #[inline]
    fn from(value: Vector<T, 2>) -> Self {
        From::from(value.to_array())
    }
}

#[cfg(feature = "mint")]
impl<T> mint::IntoMint for Vector<T, 2> {
    type MintType = mint::Vector2<T>;
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector3<T>> for Vector<T, 3> {
    #[inline]
    fn from(mint::Vector3 { x, y, z }: mint::Vector3<T>) -> Self {
        Self::new([x, y, z])
    }
}

#[cfg(feature = "mint")]
impl<T> From<Vector<T, 3>> for mint::Vector3<T> {
    #[inline]
    fn from(value: Vector<T, 3>) -> Self {
        From::from(value.to_array())
    }
}

#[cfg(feature = "mint")]
impl<T> mint::IntoMint for Vector<T, 3> {
    type MintType = mint::Vector3<T>;
}

#[cfg(feature = "mint")]
impl<T> From<mint::Vector4<T>> for Vector<T, 4> {
    #[inline]
    fn from(mint::Vector4 { x, y, z, w }: mint::Vector4<T>) -> Self {
        Self::new([x, y, z, w])
    }
}

#[cfg(feature = "mint")]
impl<T> From<Vector<T, 4>> for mint::Vector4<T> {
    #[inline]
    fn from(value: Vector<T, 4>) -> Self {
        From::from(value.to_array())
    }
}

#[cfg(feature = "mint")]
impl<T> mint::IntoMint for Vector<T, 4> {
    type MintType = mint::Vector4<T>;
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Zeroable, const N: usize> bytemuck::Zeroable for Vector<T, N> {
    #[inline]
    fn zeroed() -> Self {
        Vector::from_fn(|_| bytemuck::Zeroable::zeroed())
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Pod, const N: usize> bytemuck::Pod for Vector<T, N> {}

#[cfg(feature = "approx")]
impl<T: approx::AbsDiffEq, const N: usize> approx::AbsDiffEq for Vector<T, N>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(x, y)| x.abs_diff_eq(y, epsilon))
    }
}

#[cfg(feature = "approx")]
impl<T: approx::RelativeEq, const N: usize> approx::RelativeEq for Vector<T, N>
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
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
    }
}

#[cfg(feature = "approx")]
impl<T: approx::UlpsEq, const N: usize> approx::UlpsEq for Vector<T, N>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(x, y)| x.ulps_eq(y, epsilon, max_ulps))
    }
}

impl_eq_mint! {
    (Vector2, Vector<2>),
    (Vector3, Vector<3>),
    (Vector4, Vector<4>),
}

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize> Serialize for Vector<T, N> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            let mut struct_serializer = serializer.serialize_tuple_struct("Vector", N)?;
            for elem in self.as_slice() {
                struct_serializer.serialize_field(elem)?;
            }
            struct_serializer.end()
        } else {
            serializer.collect_seq(self.iter())
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Vector<T, N> {
    #[inline]
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ExpectedVectorData<const N: usize>;

        impl<const N: usize> de::Expected for ExpectedVectorData<N> {
            #[inline]
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "An array of {} elements", N)
            }
        }

        struct Visitor<T, const N: usize>(PhantomData<Vector<T, N>>);

        impl<'de, T: Deserialize<'de>, const N: usize> de::Visitor<'de> for Visitor<T, N> {
            type Value = Vector<T, N>;

            #[inline]
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                ExpectedVectorData::<N>.fmt(formatter)
            }

            #[inline]
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut ret_val = Vector::<T, N>::uninit();

                let mut i = 0;
                while let Some(item) = seq.next_element::<T>()? {
                    let slot = match ret_val.get_mut(i) {
                        Some(slot) => slot,
                        None => return Err(A::Error::invalid_length(i, &ExpectedVectorData::<N>)),
                    };

                    slot.write(item);
                    i += 1;
                }

                if i < N {
                    return Err(A::Error::invalid_length(i, &ExpectedVectorData::<N>));
                }

                unsafe { Ok(Vector::assume_init(ret_val)) }
            }
        }

        if deserializer.is_human_readable() {
            deserializer.deserialize_tuple_struct("Vector", 1, Visitor::<T, N>(PhantomData))
        } else {
            deserializer.deserialize_seq(Visitor::<T, N>(PhantomData))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swizzle() {
        let vector = Vector::new([1, 2, 8, 9, 3, 1, 4]);
        let swizzled = vector.swizzle(&[1, 4, 0, 0]);
        assert_eq!(swizzled.to_array(), [2, 3, 1, 1]);

        assert!(vector.try_swizzle(&[421]).is_none());
    }

    #[test]
    fn test_cross() {
        let v1 = Vector::new([3.0, 4.0, 5.0]);
        let v2 = Vector::new([7.0, 8.0, 9.0]);
        let result = Vector::cross(v1, v2);

        assert_eq!(result, Vector::new([-4.0, 8.0, -4.0]));
        #[cfg(feature = "simd")]
        {
            assert_eq!(Vector::simd_cross(v1, v2), result);
        }

        let x = Vector::X;
        let y = Vector::Y;
        let z = Vector::<f64, _>::cross(x, y);

        assert_eq!(z, Vector::Z);
        #[cfg(feature = "simd")]
        {
            assert_eq!(Vector::simd_cross(x, y), z);
        }

        let v1 = Vector::new([13.0, 24.0, 19.0]);
        let v2 = Vector::new([244.0, 190.0, 80.0]);
        let result = Vector::cross(v1, v2);

        assert_eq!(result, Vector::new([-1690.0, 3596.0, -3386.0]));
        #[cfg(feature = "simd")]
        {
            assert_eq!(Vector::simd_cross(v1, v2), result);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd() {
        use crate::simd::SimdValue;

        let vector_1 = SimdValue(Vector::new([1, 2, 4, 5]));
        let vector_2 = SimdValue(Vector::new([2, 5, 5, 3]));

        let result = vector_1 + vector_2;
        assert_eq!(result.as_slice(), &[3, 7, 9, 8]);

        let result = vector_1 - vector_2;
        assert_eq!(result.as_slice(), &[-1, -3, -1, 2]);

        let result = vector_1 * SimdValue(2);
        assert_eq!(result.as_slice(), &[2, 4, 8, 10]);
    }

    #[test]
    fn test_dot() {
        let v1 = Vector::new([1.0, 2.0, 3.0]);
        let v2 = Vector::new([7.0, 8.0, 9.0]);

        let dot = Vector::dot(v1, v2);
        assert_eq!(dot, 50.0);

        #[cfg(feature = "simd")]
        {
            let simd_dot = Vector::simd_dot(v1, v2);
            assert_eq!(dot, simd_dot);
        }

        let v = Vector::<_, 5>::X * 60.0;
        let len = v.len();
        assert_eq!(len, 60.0);
    }
}
