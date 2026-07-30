// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "simd")]
use crate::simd::{SimdAdd, SimdDiv, SimdMul, SimdSub};
#[cfg(feature = "nightly")]
use crate::{
    matrix::{Matrix, TransformHomogeneous},
    transform::{Transform, Translate},
    utils::concat,
};
use crate::{
    rotation::Rotation,
    utils::{
        array_assume_init, array_get_checked, array_get_mut_checked, array_get_unchecked,
        array_get_unchecked_mut, expand_to_copy,
        num::{
            ClosedAdd, ClosedDiv, ClosedMul, ClosedSub, One, Sqrt, Zero,
            checked::{CheckedDiv, CheckedMul},
        },
        shrink_to, zip_map,
    },
    vector::Vector,
};
#[cfg(feature = "serde")]
use core::marker::PhantomData;
#[cfg(feature = "simd")]
use core::simd::{Simd, SimdElement};
use core::{
    array::{self, IntoIter},
    borrow::{Borrow, BorrowMut},
    fmt,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    ptr,
    slice::{self, Iter, IterMut},
};
#[cfg(feature = "serde")]
use serde_core::{
    de::{self, Deserialize, Deserializer, Error, Expected, SeqAccess},
    ser::{Serialize, SerializeTupleStruct, Serializer},
};

/// A type which represents a position in space. This type is generic over its dimension,
/// and the type of its scalar components.
#[repr(C)]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Point<T = f32, const N: usize = 3> {
    data: [T; N],
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for Point<T, N> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.write_str("Point ")?;
        let mut debug_list = fmtr.debug_list();
        debug_list.entries(self.data.iter());
        debug_list.finish()
    }
}

impl<T: Default, const N: usize> Default for Point<T, N> {
    #[inline]
    fn default() -> Self {
        Self {
            data: array::from_fn(|_| Default::default()),
        }
    }
}

pub type Point0<T = f32> = Point<T, 0>;
pub type Point1<T = f32> = Point<T, 1>;
pub type Point2<T = f32> = Point<T, 2>;
pub type Point3<T = f32> = Point<T, 3>;
pub type Point4<T = f32> = Point<T, 4>;

impl_coerce_to_fields! {
    Point<{T, 1}> => X,
    Point<{T, 2}> => Xy,
    Point<{T, 3}> => Xyz,
    Point<{T, 4}> => Xyzw,
}

impl<T: Copy, const N: usize> Point<T, N> {
    /// Create a new `Point` where each field is set to the given `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let point = Point::<_, 64>::splat(127i8);
    ///
    /// for element in point {
    ///     assert_eq!(element, 127);
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub const fn splat(value: T) -> Self {
        Self { data: [value; N] }
    }

    /// Expand the given `Point` into a larger dimension, extending new fields with the given `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let point = Point::new([1, 2, 3]);
    /// let expanded: Point::<_, 6> = point.expand_to::<6>(6);
    ///
    /// assert_eq!(&expanded, &[1, 2, 3, 6, 6, 6]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn expand_to<const N1: usize>(self, value: T) -> Point<T, N1> {
        Point {
            data: expand_to_copy(self.data, value),
        }
    }
}

impl<T: Zero, const N: usize> Zero for Point<T, N> {
    const ZERO: Self = Self::new(Zero::ZERO);
}

impl<T: Zero, const N: usize> Point<T, N> {
    /// Returns a new `Point` at the origin, where each field is set to `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let point = Point::<f32, 3>::origin();
    /// assert_eq!(&point, &[0.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn origin() -> Self {
        Self::ZERO
    }
}

impl<T, const N: usize> Point<T, N> {
    /// The dimension of the `Point`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// type Point45<T> = Point::<T, 45>;
    ///
    /// assert_eq!(<Point45<i32>>::LENGTH, 45);
    /// ```
    pub const LENGTH: usize = N;

    /// Create a new `Point` from the given `array`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let point = Point::new([1, 2, 3, 5]);
    /// assert_eq!(&point, &[1, 2, 3, 5]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(array: [T; N]) -> Self {
        Self { data: array }
    }

    /// Create a new `Point` from the given function `f`.
    ///
    /// The function takes the index of the field it is initializing.
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let point = Point::<usize, 5>::from_fn(|i| i * 2);
    ///
    /// assert_eq!(&point, &[0, 2, 4, 6, 8]);
    /// ```
    #[must_use]
    #[inline]
    pub fn from_fn<F: FnMut(usize) -> T>(f: F) -> Self {
        Self::new(array::from_fn(f))
    }

    /// Returns a `Point` containtaining uninitialized data.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let point = Point::<f64, 4>::uninit();
    /// ```
    #[must_use]
    #[inline]
    pub const fn uninit() -> Point<MaybeUninit<T>, N> {
        Point {
            data: [const { MaybeUninit::uninit() }; N],
        }
    }

    #[must_use]
    #[inline]
    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> Point<U, N> {
        Point {
            data: self.data.map(f),
        }
    }

    #[must_use]
    #[inline]
    pub const fn each_ref(&self) -> Point<&T, N> {
        Point {
            data: self.data.each_ref(),
        }
    }

    #[must_use]
    #[inline]
    pub const fn each_mut(&mut self) -> Point<&mut T, N> {
        Point {
            data: self.data.each_mut(),
        }
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

    /// Returns a const pointer to the start of the `Point` data.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    /// use std::ptr;
    ///
    /// let point = Point::new([1u64, 2, 4, 8, 24]);
    ///
    /// let ptr: *const u64 = point.as_ptr();
    /// assert!(ptr::eq(ptr, &point[0]));
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the start of the `Point` data.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    /// use std::ptr;
    ///
    /// let mut point = Point::new([1u64, 2, 4, 8, 24]);
    ///
    /// let ptr: *mut u64 = point.as_mut_ptr();
    /// assert!(ptr::eq(ptr, &point[0]));
    ///
    /// unsafe {
    ///     ptr.add(2).write(8);
    /// }
    ///
    /// assert_eq!(&point, &[1, 2, 8, 8, 24]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Access the `Point` as a `&[T]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let point = Point::new([1, 2, 4, 8, 14, 2]);
    /// let slice: &[i16] = point.as_slice();
    ///
    /// assert_eq!(slice, &[1, 2, 4, 8, 14, 2]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), N) }
    }

    /// Access the `Point` mutably as a `&mut [T]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let mut point = Point::new([1, 2, 4, 8, 14, 2]);
    /// let slice: &mut [i16] = point.as_mut_slice();
    ///
    /// slice[1] = 14;
    ///
    /// assert_eq!(&point, &[1, 14, 4, 8, 14, 2]);
    /// ``` 
    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), N) }
    }

    #[deprecated(note = "use Vector::to_array instead")]
    #[must_use]
    #[inline]
    pub const fn into_array(self) -> [T; N] {
        let array = unsafe { ptr::read(&self.data) };
        let _self = ManuallyDrop::new(self);
        array
    }

    /// Convert the given `point` into a fixed size array.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point4;
    ///
    /// let point = Point4::splat(18);
    /// let array = point.to_array();
    ///
    /// assert_eq!(array, [18; 4]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn to_array(self) -> [T; N] {
        let array = unsafe { ptr::read(&self.data) };
        let _this = ManuallyDrop::new(self);
        array
    }

    /// Returns the distance to the `other` `Point`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let origin = Point::<f64, 3>::origin();
    /// let other = Point::new([3.0, 4.0, 0.0]);
    ///
    /// assert_eq!(origin.distance_to(other), 5.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn distance_to<U: Sub<T>>(self, other: Point<U, N>) -> U::Output
    where
        U::Output: ClosedMul + Copy + ClosedAdd + Zero + Sqrt,
    {
        self.vector_to(other).len()
    }

    /// Returns the distance squared to the `other` `Point`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let origin = Point::<f64, 3>::origin();
    /// let other = Point::new([3.0, 4.0, 0.0]);
    ///
    /// assert_eq!(other.distance_squared_to(origin), 25.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn distance_squared_to<U: Sub<T>>(self, other: Point<U, N>) -> U::Output
    where
        U::Output: ClosedMul + Copy + ClosedAdd + Zero,
    {
        self.vector_to(other).len_squared()
    }

    /// Returns the `Vector` which would translate `self` to `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::{point::Point, vector::Vector};
    /// 
    /// let p1 = Point::new([1.0, 2.0, 6.0]);
    /// let p2 = Point::new([3.0, 1.0, 8.0]);
    ///
    /// assert_eq!(p1.vector_to(p2), Vector::new([2.0, -1.0, 2.0]));
    /// ```
    #[must_use]
    #[inline]
    pub fn vector_to<U: Sub<T>>(self, other: Point<U, N>) -> Vector<U::Output, N> {
        other - self
    }

    /// Returns the normalized direction vector which points from `self` to `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::{point::Point, vector::Vector};
    /// 
    /// let direction = Point::<f64, 3>::origin().direction_to(Point::new([0.0, 0.7, 0.0]));
    /// assert_eq!(direction, Vector::Y);
    /// ```
    #[must_use]
    #[inline]
    pub fn direction_to<U: Sub<T>>(self, other: Point<U, N>) -> Vector<U::Output, N>
    where
        U::Output: ClosedMul + Copy + ClosedAdd + ClosedDiv + Zero + Sqrt,
    {
        Vector::normalized(self.vector_to(other))
    }

    /// Converts the `Point` into a `Vector` with identical fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::{point::Point, vector::Vector};
    ///
    /// let point = Point::new([1, 5, 2, 9, 135, 2104]);
    /// let vector: Vector<i32, _> = point.to_vector();
    /// assert_eq!(&vector, &[1, 5, 2, 9, 135, 2104]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn to_vector(self) -> Vector<T, N> {
        Vector::new(self.to_array())
    }

    #[deprecated = "use Point::to_vector instead"]
    #[must_use]
    #[inline]
    pub const fn into_vector(self) -> Vector<T, N> {
        Vector::new(self.to_array())
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub fn shrink(self) -> Point<T, { N - 1 }> {
        Point {
            data: shrink_to(self.data),
        }
    }

    #[must_use]
    #[inline]
    pub fn shrink_to<const N_NEW: usize>(self) -> Point<T, N_NEW> {
        Point {
            data: shrink_to(self.data),
        }
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn expand(self, to_append: T) -> Point<T, { N + 1 }> {
        let data = unsafe { ptr::read(&self.data) };
        let _self = ManuallyDrop::new(self);
        Point {
            data: concat(data, [to_append]),
        }
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T: Copy + One + ClosedAdd + ClosedDiv + ClosedSub, const N: usize> Point<T, N> {
    /// Returns the point exactly between `self` and `other`.
    ///
    /// This method should not overflow.
    ///
    /// # Example
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let p1 = Point::new([2.0; 3]);
    /// let p2 = Point::new([4.0; 3]);
    ///
    /// assert_eq!(p1.midpoint(p2), Point::new([3.0; 3]));
    /// ```
    #[must_use]
    #[inline]
    pub fn midpoint(self, other: Point<T, N>) -> Point<T, N> {
        let mut v = self.vector_to(other);
        v = v / (T::ONE + T::ONE);
        self + v
    }
}

impl<T: Copy + ClosedDiv + ClosedSub + ClosedMul + ClosedAdd + Zero, const N: usize> Point<T, N> {
    /// Returns the `Point` rotated around `center_of_rotation`, by `rotation`.
    ///
    /// # Example
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use vectral::{point::Point, vector::Vector, quaternion::Quaternion, rotation::angle::Degrees};
    ///
    /// let point = Point::new([1.0, 0.0, 0.0]);
    /// let origin = Point::origin();
    ///
    /// let rotation = Quaternion::from_angle_axis(Degrees(90.0), Vector::Y);
    ///
    /// assert_relative_eq!(point.rotated_around(origin, rotation), Point::new([0.0, 0.0, -1.0]));
    /// ```
    #[must_use]
    #[inline]
    pub fn rotated_around<R: Rotation<N, Scalar = T>>(
        self,
        center_of_rotation: Self,
        rotation: R,
    ) -> Self {
        let dir = self.vector_to(center_of_rotation);
        let rotated_dir = rotation.transform_vector(dir);
        self + dir - rotated_dir
    }
}

impl<T, const N: usize> Point<MaybeUninit<T>, N> {
    /// Initialize a `Point` of `MaybeUninit` data, assuming that
    /// each field is initialized.
    ///
    /// # Safety
    ///
    /// This method must only be called on a `Point` where each field
    /// of the `Point` has been initialized through the `MaybeUninit`,
    /// otherwise this method may allow access to uninitialized memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::point::Point;
    ///
    /// let mut point = Point::<usize, 5>::uninit();
    ///
    /// for (i, item) in point.iter_mut().enumerate() {
    ///     unsafe {
    ///         item.write(i + 200);
    ///     }
    /// }
    ///
    /// let point = unsafe {
    ///     Point::assume_init(point)
    /// };
    ///
    /// assert_eq!(&point, &[200, 201, 202, 203, 204]);
    /// ```
    #[must_use]
    #[inline]
    pub unsafe fn assume_init(self) -> Point<T, N> {
        Point {
            data: unsafe { array_assume_init(self.data) },
        }
    }
}

#[cfg(feature = "nightly")]
impl<T, const N: usize> TransformHomogeneous<N> for Point<T, N>
where
    T: Zero + One + PartialEq + Copy + DivAssign + ClosedMul + ClosedAdd,
    Matrix<T, { N + 1 }, { N + 1 }>: Sized,
{
    type Scalar = T;

    #[inline]
    fn transform_homogeneous(self, matrix: Matrix<Self::Scalar, { N + 1 }, { N + 1 }>) -> Self {
        let vec = self.to_vector().transform_homogeneous(matrix);
        vec.to_point()
    }
}

impl<T, const N: usize> Index<usize> for Point<T, N> {
    type Output = T;
    #[track_caller]
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        Index::index(&self.data, index)
    }
}

impl<T, const N: usize> IndexMut<usize> for Point<T, N> {
    #[track_caller]
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.data, index)
    }
}

impl<T, const N: usize> From<Vector<T, N>> for Point<T, N> {
    #[inline]
    fn from(value: Vector<T, N>) -> Self {
        Self {
            data: value.to_array(),
        }
    }
}

impl<T: Mul<U>, U: Copy, const N: usize> Mul<U> for Point<T, N> {
    type Output = Point<T::Output, N>;
    #[inline]
    fn mul(self, rhs: U) -> Self::Output {
        self.map(|elem| elem * rhs)
    }
}

impl<T: Copy + CheckedMul, const N: usize> CheckedMul<T> for Point<T, N> {
    #[inline]
    fn checked_mul(self, rhs: T) -> Option<Self::Output> {
        let mut v: Point<_, N> = Point::uninit();

        for (i, elem) in self.into_iter().enumerate() {
            let res = elem.checked_mul(rhs)?;
            unsafe {
                v.get_unchecked_mut(i).write(res);
            }
        }

        unsafe { Some(Point::assume_init(v)) }
    }
}

impl<T: MulAssign<U>, U: Copy, const N: usize> MulAssign<U> for Point<T, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: U) {
        for elem in &mut self.data {
            elem.mul_assign(rhs);
        }
    }
}

impl<T: Div<U>, U: Copy, const N: usize> Div<U> for Point<T, N> {
    type Output = Point<T::Output, N>;
    #[inline]
    fn div(self, rhs: U) -> Self::Output {
        self.map(|elem| elem / rhs)
    }
}

impl<T: CheckedDiv<U>, U: Copy, const N: usize> CheckedDiv<U> for Point<T, N> {
    #[inline]
    fn checked_div(self, rhs: U) -> Option<Self::Output> {
        let mut v: Point<_, N> = Point::uninit();

        for (i, elem) in self.into_iter().enumerate() {
            let res = elem.checked_div(rhs)?;
            unsafe {
                v.get_unchecked_mut(i).write(res);
            }
        }

        unsafe { Some(Point::assume_init(v)) }
    }
}

impl<T: DivAssign<U>, U: Copy, const N: usize> DivAssign<U> for Point<T, N> {
    #[inline]
    fn div_assign(&mut self, rhs: U) {
        for elem in &mut self.data {
            elem.div_assign(rhs);
        }
    }
}

impl<T: Add<U>, U, const N: usize> Add<Vector<U, N>> for Point<T, N> {
    type Output = Point<T::Output, N>;
    #[inline]
    fn add(self, rhs: Vector<U, N>) -> Self::Output {
        Point::new(zip_map(self.data, rhs.to_array(), Add::add))
    }
}

impl<T: AddAssign<U>, U, const N: usize> AddAssign<Vector<U, N>> for Point<T, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Vector<U, N>) {
        for (l, r) in self.data.iter_mut().zip(rhs) {
            l.add_assign(r);
        }
    }
}

impl<T: Sub<U>, U, const N: usize> Sub<Vector<U, N>> for Point<T, N> {
    type Output = Point<T::Output, N>;
    #[inline]
    fn sub(self, rhs: Vector<U, N>) -> Self::Output {
        Point::new(zip_map(self.data, rhs.to_array(), Sub::sub))
    }
}

impl<T: SubAssign<U>, U, const N: usize> SubAssign<Vector<U, N>> for Point<T, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Vector<U, N>) {
        for (l, r) in self.data.iter_mut().zip(rhs) {
            l.sub_assign(r);
        }
    }
}

impl<T: Sub<U>, U, const N: usize> Sub<Point<U, N>> for Point<T, N> {
    type Output = Vector<T::Output, N>;
    #[inline]
    fn sub(self, rhs: Point<U, N>) -> Self::Output {
        Vector::new(zip_map(self.data, rhs.to_array(), Sub::sub))
    }
}

impl<T: SubAssign<U>, U, const N: usize> SubAssign<Point<U, N>> for Point<T, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Point<U, N>) {
        for (l, r) in self.data.iter_mut().zip(rhs) {
            l.sub_assign(r);
        }
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdMul<T> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedMul,
{
    type Output = Self;
    fn simd_mul(self, rhs: T) -> Self::Output {
        let rhs = Simd::splat(rhs);
        let lhs = Simd::from_array(self.to_array());
        (lhs * rhs).into()
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdDiv<T> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedDiv,
{
    type Output = Self;
    fn simd_div(self, rhs: T) -> Self::Output {
        let rhs = Simd::splat(rhs);
        let lhs = Simd::from_array(self.to_array());
        (lhs / rhs).into()
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdAdd<T> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedAdd,
{
    type Output = Self;
    fn simd_add(self, rhs: T) -> Self::Output {
        let rhs = Simd::splat(rhs);
        let lhs = Simd::from_array(self.to_array());
        (lhs + rhs).into()
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdSub<T> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedSub,
{
    type Output = Self;
    fn simd_sub(self, rhs: T) -> Self::Output {
        let rhs = Simd::splat(rhs);
        let lhs = Simd::from_array(self.to_array());
        (lhs - rhs).into()
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdMul<Vector<T, N>> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedMul,
{
    type Output = Self;
    fn simd_mul(self, rhs: Vector<T, N>) -> Self::Output {
        let rhs = Simd::from_array(rhs.to_array());
        let lhs = Simd::from_array(self.to_array());
        (lhs * rhs).into()
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdDiv<Vector<T, N>> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedDiv,
{
    type Output = Self;
    fn simd_div(self, rhs: Vector<T, N>) -> Self::Output {
        let rhs = Simd::from_array(rhs.to_array());
        let lhs = Simd::from_array(self.to_array());
        (lhs / rhs).into()
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdAdd<Vector<T, N>> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedAdd,
{
    type Output = Self;
    fn simd_add(self, rhs: Vector<T, N>) -> Self::Output {
        let rhs = Simd::from_array(rhs.to_array());
        let lhs = Simd::from_array(self.to_array());
        (lhs + rhs).into()
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdSub<Vector<T, N>> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedSub,
{
    type Output = Self;
    fn simd_sub(self, rhs: Vector<T, N>) -> Self::Output {
        let rhs = Simd::from_array(rhs.to_array());
        let lhs = Simd::from_array(self.to_array());
        (lhs - rhs).into()
    }
}

impl<T, const N: usize> AsRef<[T; N]> for Point<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.data
    }
}

impl<T, const N: usize> AsMut<[T; N]> for Point<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }
}

impl<T, const N: usize> Borrow<[T; N]> for Point<T, N> {
    #[inline]
    fn borrow(&self) -> &[T; N] {
        &self.data
    }
}

impl<T, const N: usize> BorrowMut<[T; N]> for Point<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }
}

impl<T, const N: usize> From<Point<T, N>> for [T; N] {
    #[inline]
    fn from(value: Point<T, N>) -> Self {
        value.data
    }
}

impl<T, const N: usize> From<[T; N]> for Point<T, N> {
    #[inline]
    fn from(value: [T; N]) -> Self {
        Self { data: value }
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for Point<T, N> {
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        PartialEq::eq(self.as_slice(), other)
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U]> for Point<T, N> {
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        PartialEq::eq(self.as_slice(), other)
    }
}

#[cfg(feature = "nightly")]
impl<T, const DIM: usize> Translate<DIM> for Point<T, DIM>
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

impl<T, const N: usize> IntoIterator for Point<T, N> {
    type IntoIter = IntoIter<T, N>;
    type Item = T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Point<T, N> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Point<T, N> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Zeroable, const N: usize> bytemuck::Zeroable for Point<T, N> {
    #[inline]
    fn zeroed() -> Self {
        Point::from_fn(|_| bytemuck::Zeroable::zeroed())
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Pod, const N: usize> bytemuck::Pod for Point<T, N> {}

#[cfg(feature = "mint")]
impl<T> From<mint::Point2<T>> for Point<T, 2> {
    #[inline]
    fn from(mint::Point2 { x, y }: mint::Point2<T>) -> Self {
        Self::new([x, y])
    }
}

#[cfg(feature = "mint")]
impl<T> From<Point<T, 2>> for mint::Point2<T> {
    #[inline]
    fn from(value: Point<T, 2>) -> Self {
        From::from(value.to_array())
    }
}

#[cfg(feature = "mint")]
impl<T> mint::IntoMint for Point<T, 2> {
    type MintType = mint::Point2<T>;
}

#[cfg(feature = "mint")]
impl<T> From<mint::Point3<T>> for Point<T, 3> {
    #[inline]
    fn from(mint::Point3 { x, y, z }: mint::Point3<T>) -> Self {
        Self::new([x, y, z])
    }
}

#[cfg(feature = "mint")]
impl<T> From<Point<T, 3>> for mint::Point3<T> {
    #[inline]
    fn from(value: Point<T, 3>) -> Self {
        From::from(value.to_array())
    }
}

#[cfg(feature = "mint")]
impl<T> mint::IntoMint for Point<T, 3> {
    type MintType = mint::Point3<T>;
}

#[cfg(feature = "approx")]
impl<T: approx::AbsDiffEq, const N: usize> approx::AbsDiffEq for Point<T, N>
where
    T::Epsilon: Clone,
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
            .all(|(x, y)| x.abs_diff_eq(y, epsilon.clone()))
    }
}

#[cfg(feature = "approx")]
impl<T: approx::RelativeEq, const N: usize> approx::RelativeEq for Point<T, N>
where
    T::Epsilon: Clone,
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
            .all(|(x, y)| x.relative_eq(y, epsilon.clone(), max_relative.clone()))
    }
}

#[cfg(feature = "approx")]
impl<T: approx::UlpsEq, const N: usize> approx::UlpsEq for Point<T, N>
where
    T::Epsilon: Clone,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(x, y)| x.ulps_eq(y, epsilon.clone(), max_ulps))
    }
}

impl_eq_mint! {
    (Point2, Point<2>),
    (Point3, Point<3>),
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> From<Simd<T, N>> for Point<T, N> {
    #[inline]
    fn from(value: Simd<T, N>) -> Self {
        Self::new(value.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> From<Point<T, N>> for Simd<T, N> {
    #[inline]
    fn from(value: Point<T, N>) -> Self {
        Self::from_array(value.to_array())
    }
}

#[cfg(feature = "serde")]
impl<T: Serialize, const N: usize> Serialize for Point<T, N> {
    #[inline]
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            let mut struct_serializer = serializer.serialize_tuple_struct("Point", N)?;
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
impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for Point<T, N> {
    #[inline]
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ExpectedPointData<const N: usize>;

        impl<const N: usize> de::Expected for ExpectedPointData<N> {
            #[inline]
            fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "An array of {} elements", N)
            }
        }

        struct Visitor<T, const N: usize>(PhantomData<Point<T, N>>);

        impl<'de, T: Deserialize<'de>, const N: usize> de::Visitor<'de> for Visitor<T, N> {
            type Value = Point<T, N>;

            #[inline]
            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                ExpectedPointData::<N>.fmt(formatter)
            }

            #[inline]
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut ret_val = Point::<T, N>::uninit();

                let mut i = 0;
                while let Some(item) = seq.next_element::<T>()? {
                    let slot = match ret_val.get_mut(i) {
                        Some(slot) => slot,
                        None => return Err(A::Error::invalid_length(i, &ExpectedPointData::<N>)),
                    };

                    slot.write(item);
                    i += 1;
                }

                if i < N {
                    return Err(A::Error::invalid_length(i, &ExpectedPointData::<N>));
                }

                unsafe { Ok(Point::assume_init(ret_val)) }
            }
        }

        if deserializer.is_human_readable() {
            deserializer.deserialize_tuple_struct("Point", 1, Visitor::<T, N>(PhantomData))
        } else {
            deserializer.deserialize_seq(Visitor::<T, N>(PhantomData))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vector3;
    use approx::assert_relative_eq;

    #[test]
    fn test_point_direction() {
        let p0 = Point3::<f32>::ZERO;
        let p1 = Point3::new([1.0f32, 0.0, 0.0]);

        assert_eq!(p0.vector_to(p1), Vector3::new([1.0, 0.0, 0.0]));
        assert_eq!(p1.vector_to(p0), Vector3::new([-1.0, 0.0, 0.0]));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde() {
        let point = Point::new([23.0, 45.0, 128.0, 1421.014, 291.0]);

        let point_string = serde_json::to_string(&point).unwrap();
        let point_deserialized = serde_json::from_str(&point_string).unwrap();

        assert_relative_eq!(&point, &point_deserialized);

        let point_data = rmp_serde::to_vec(&point).unwrap();
        let point_deserialized = rmp_serde::from_slice(&point_data).unwrap();

        assert_relative_eq!(&point, &point_deserialized);
    }
}
