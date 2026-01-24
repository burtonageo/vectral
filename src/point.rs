#[cfg(feature = "nightly")]
use crate::{
    matrix::{Matrix, TransformHomogeneous},
    transform::{Transform, Translate},
    utils::{expand, shrink, num::One},
};
#[cfg(feature = "simd")]
use crate::{
    simd::{SimdAdd, SimdDiv, SimdMul, SimdSub},
    utils::num::ClosedDiv,
};
use crate::{
    utils::{
        array_get_checked, array_get_mut_checked, array_get_unchecked, array_get_unchecked_mut,
    },
    utils::{
        expand_to,
        num::{
            ClosedAdd, ClosedMul, ClosedSub, Sqrt, Zero,
            checked::{CheckedDiv, CheckedMul},
        },
        shrink_to, zip_map,
    },
    vector::Vector,
};
#[cfg(feature = "serde")]
use core::marker::PhantomData;
#[cfg(feature = "simd")]
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
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
    #[must_use]
    #[inline]
    pub const fn splat(value: T) -> Self {
        Self { data: [value; N] }
    }

    #[must_use]
    #[inline]
    pub const fn expand_to<const N1: usize>(self, value: T) -> Point<T, N1> {
        Point {
            data: expand_to(self.data, value),
        }
    }
}

impl<T: Zero, const N: usize> Zero for Point<T, N> {
    const ZERO: Self = Self::new(Zero::ZERO);
}

impl<T, const N: usize> Point<T, N> {
    pub const LENGTH: usize = N;

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
        let array = unsafe { ptr::read(&self.data) };
        let _self = ManuallyDrop::new(self);
        array
    }

    #[must_use]
    #[inline]
    pub const fn to_array(self) -> [T; N] {
        let array = unsafe { ptr::read(&self.data) };
        let _this = ManuallyDrop::new(self);
        array
    }

    #[must_use]
    #[inline]
    pub fn distance_to<U: Sub<T>>(self, other: Point<U, N>) -> U::Output
    where
        U::Output: ClosedMul + Copy + ClosedAdd + Zero + Sqrt,
    {
        self.vector_to(other).len()
    }

    #[must_use]
    #[inline]
    pub fn vector_to<U: Sub<T>>(self, other: Point<U, N>) -> Vector<U::Output, N> {
        other - self
    }

    #[must_use]
    #[inline]
    pub fn direction_to<U: Sub<T>>(self, other: Point<U, N>) -> Vector<U::Output, N>
    where
        U::Output: ClosedMul + Copy + ClosedAdd + CheckedDiv<Output = U::Output> + Zero + Sqrt,
    {
        Vector::normalized(self.vector_to(other))
    }

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
            data: shrink(self.data),
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
            data: expand(data, to_append),
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

impl<T: PartialOrd + ClosedSub, const N: usize> Point<T, N> {
    #[must_use]
    #[inline]
    pub fn is_nearly_equal(self, to: Point<T, N>, epsilon: T) -> bool {
        (self - to).into_iter().all(|elem| elem < epsilon)
    }
}

impl<T, const N: usize> Point<MaybeUninit<T>, N> {
    #[must_use]
    #[inline]
    pub unsafe fn assume_init(self) -> Point<T, N> {
        Point {
            data: unsafe { MaybeUninit::array_assume_init(self.data) },
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
        let vec = self.into_vector().transform_homogeneous(matrix);
        vec.into_point()
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
        for (l, r) in self.data.iter_mut().zip(rhs.into_iter()) {
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
        for (l, r) in self.data.iter_mut().zip(rhs.into_iter()) {
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
        for (l, r) in self.data.iter_mut().zip(rhs.into_iter()) {
            l.sub_assign(r);
        }
    }
}

#[cfg(feature = "simd")]
impl<T, const N: usize> SimdMul<T> for Point<T, N>
where
    T: SimdElement,
    Simd<T, N>: ClosedMul,
    LaneCount<N>: SupportedLaneCount,
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
    LaneCount<N>: SupportedLaneCount,
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
    LaneCount<N>: SupportedLaneCount,
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
    LaneCount<N>: SupportedLaneCount,
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
    LaneCount<N>: SupportedLaneCount,
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
    LaneCount<N>: SupportedLaneCount,
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
    LaneCount<N>: SupportedLaneCount,
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
    LaneCount<N>: SupportedLaneCount,
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
impl<T: approx::RelativeEq, const N: usize> approx::RelativeEq for Point<T, N>
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
impl<T: approx::UlpsEq, const N: usize> approx::UlpsEq for Point<T, N>
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
    (Point2, Point<2>),
    (Point3, Point<3>),
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> From<Simd<T, N>> for Point<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    #[inline]
    fn from(value: Simd<T, N>) -> Self {
        Self::new(value.to_array())
    }
}

#[cfg(feature = "simd")]
impl<T: SimdElement, const N: usize> From<Point<T, N>> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
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

    #[test]
    fn test_point_direction() {
        let p0 = Point3::<f32>::ZERO;
        let p1 = Point3::new([1.0f32, 0.0, 0.0]);

        assert_eq!(p0.vector_to(p1), Vector3::new([1.0, 0.0, 0.0]));
        assert_eq!(p1.vector_to(p0), Vector3::new([-1.0, 0.0, 0.0]));
    }
}
