// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "simd")]
use crate::simd::SimdMul;
#[cfg(feature = "nightly")]
use crate::{
    point::Point3,
    rotation::Rotation,
    utils::{flatten, num::checked::CheckedDiv, shrink_to, shrink_to_copy},
    vector::Vector4,
};
use crate::{
    rotation::{angle::Angle, quaternion::Quaternion},
    utils::{
        array_get_checked, array_get_mut_checked, array_get_unchecked, array_get_unchecked_mut,
        num::{
            Abs, Bounded, ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub, One, Sqrt, Trig,
            Zero, checked::CheckedAddAssign, n,
        },
        sum, zip_map,
    },
    vector::{Vector, Vector3},
};
#[cfg(feature = "simd")]
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
#[cfg(feature = "nightly")]
use core::{
    array,
    cmp::{Ordering, max_by},
};
use core::{
    borrow::{Borrow, BorrowMut},
    convert::{AsMut, AsRef},
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    ptr, slice,
};

#[cfg(test)]
mod tests;

/// A row-major matrix of arbitrary dimensions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Matrix<T = f32, const ROWS: usize = 4, const COLS: usize = 4> {
    data: [[T; COLS]; ROWS],
}

impl<T: Default, const ROWS: usize, const COLS: usize> Default for Matrix<T, ROWS, COLS> {
    #[inline]
    fn default() -> Self {
        Self::from_fn(|_, _| Default::default())
    }
}

impl<T, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    pub const NUM_ELEMENTS: usize = ROWS * COLS;

    /// Create a new `Matrix` from the given nested array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let data: [[i32; 2]; 2] = [[1, 2], [3, 4]];
    /// let matrix: Matrix<i32, 2, 2> = Matrix::new(data);
    /// # let _matrix = matrix;
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(data: [[T; COLS]; ROWS]) -> Self {
        Self { data }
    }

    #[must_use]
    #[inline]
    pub fn from_fn<F: FnMut(usize, usize) -> T>(mut f: F) -> Self {
        let mut mat = Matrix::uninit();

        let mut row = 0;
        while row < ROWS {
            let mut col = 0;
            while col < COLS {
                unsafe {
                    mat.get_unchecked_mut(row, col).write(f(row, col));
                }

                col += 1;
            }
            row += 1;
        }

        unsafe { Matrix::assume_init(mat) }
    }

    /// Creates a new `Matrix`, where every element is uninitialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// # use std::mem::MaybeUninit;
    /// let matrix = Matrix::uninit();
    /// # let _mat: Matrix<MaybeUninit<i32>, 4, 4> = matrix;
    /// ```
    #[must_use]
    #[inline]
    pub const fn uninit() -> Matrix<MaybeUninit<T>, ROWS, COLS> {
        unsafe { mem::transmute_copy(&MaybeUninit::<Matrix<T, ROWS, COLS>>::uninit()) }
    }

    /// Returns a reference to the inner array of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    ///
    /// let array: &[[i32; 3]; 2] = matrix.as_array();
    /// assert_eq!(array[0], [1, 2, 3]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_array(&self) -> &[[T; COLS]; ROWS] {
        &self.data
    }

    #[deprecated(note = "use Matrix::to_array instead")]
    #[must_use]
    #[inline]
    pub const fn into_array(self) -> [[T; COLS]; ROWS] {
        self.to_array()
    }

    #[must_use]
    #[inline]
    pub const fn to_array(self) -> [[T; COLS]; ROWS] {
        let array = unsafe { ptr::read(&self.data) };
        let _self = ManuallyDrop::new(self);
        array
    }

    /// Returns a mutable reference to the inner array of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let mut matrix = Matrix::new([
    ///     [0.0, 1.0],
    ///     [2.0, 3.0],
    ///     [4.0, 5.0],
    /// ]);
    ///
    /// let mut array: &mut [[f64; 2]; 3] = matrix.as_array_mut();
    /// array[0][0] = 64.0;
    ///
    /// assert_eq!(matrix[0][0], 64.0);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_array_mut(&mut self) -> &mut [[T; COLS]; ROWS] {
        &mut self.data
    }

    /// Returns a reference to the inner array of the matrix as a flattened slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    ///
    /// assert_eq!(matrix.as_slice(), &[1, 2, 3, 4, 5, 6]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), ROWS * COLS) }
    }

    /// Access the `Matrix` as flat array of all elements contained in the `Matrix`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let matrix = Matrix::new([
    ///     [1, 2],
    ///     [3, 4],
    /// ]);
    ///
    /// let array: &'_ [i32; 4] = matrix.as_flattened_array();
    ///
    /// assert_eq!(array, &[1, 2, 3, 4]);
    /// ```
    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn as_flattened_array(&self) -> &[T; ROWS * COLS]
    where
        [T; ROWS * COLS]: Sized,
    {
        unsafe { &*(self as *const _ as *const [T; ROWS * COLS]) }
    }

    /// Returns a mutable reference to the inner array of the matrix as a flattened slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let mut matrix = Matrix::new([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    ///
    /// let mut slice = matrix.as_mut_slice();
    /// slice[4] = 9;
    ///
    /// assert_eq!(matrix, Matrix::new([
    ///     [1, 2, 3],
    ///     [4, 9, 6],
    /// ]));
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), ROWS * COLS) }
    }

    /// Access the `Matrix` mutably as flat array of all elements contained in the `Matrix`.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let mut matrix = Matrix::new([
    ///     [1, 2],
    ///     [3, 4],
    /// ]);
    ///
    /// let array: &'_ mut [i32; 4] = matrix.as_mut_flattened_array();
    ///
    /// assert_eq!(array, &[1, 2, 3, 4]);
    /// array[2] = 5;
    ///
    /// assert_eq!(&matrix, &Matrix::new([
    ///     [1, 2],
    ///     [5, 4],
    /// ]));
    /// ```
    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn as_mut_flattened_array(&mut self) -> &mut [T; ROWS * COLS]
    where
        [T; ROWS * COLS]: Sized,
    {
        unsafe { &mut *(self as *mut _ as *mut [T; ROWS * COLS]) }
    }

    /// Access the start of the `Matrix`'s element data as a pointer.
    ///
    /// # Example
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let matrix = Matrix::new([
    ///     [5, 6, 4, 2],
    ///     [1, 1, 3, 4],
    ///     [2, 7, 9, 0],
    /// ]);
    ///
    /// let ptr = matrix.as_ptr();
    /// unsafe {
    ///     assert_eq!(*ptr, 5);
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr().cast()
    }

    /// Access the start of the `Matrix`'s element data as a mutable pointer.
    ///
    /// # Example
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let mut matrix = Matrix::new([
    ///     [5, 6, 4, 2],
    ///     [1, 1, 3, 4],
    ///     [2, 7, 9, 0],
    /// ]);
    ///
    /// let ptr = matrix.as_mut_ptr();
    /// unsafe {
    ///     assert_eq!(*ptr, 5);
    ///     *ptr = 31;
    /// }
    ///
    /// assert_eq!(matrix[0][0], 31);
    /// ```
    #[must_use]
    #[inline]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr().cast()
    }

    /// Attempt to get a reference to the element at `Matrix[row][col]`.
    ///
    /// This method returns `None` if either of the given indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let matrix: Matrix<f32, _, _> = Matrix::new([
    ///     [5.0, 6.0, 4.0, 2.0],
    ///     [1.0, 1.0, 3.0, 4.0],
    ///     [2.0, 7.0, 9.0, 0.0],
    ///     [0.0, 0.0, 0.0, 1.0],
    /// ]);
    ///
    /// let elem = matrix.get(1, 1);
    /// assert_eq!(elem, Some(&1.0));
    ///
    /// let elem = matrix.get(6, 3);
    /// assert_eq!(elem, None);
    /// ```
    #[must_use]
    #[inline]
    pub const fn get(&self, row: usize, col: usize) -> Option<&T> {
        match array_get_checked(&self.data, row) {
            Some(row) => array_get_checked(row, col),
            None => None,
        }
    }

    /// Attempt to get a mutable reference to the element at `Matrix[row][col]`.
    ///
    /// This method returns `None` if either of the given indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let mut matrix: Matrix<f32, _, _> = Matrix::new([
    ///     [5.0, 6.0, 4.0, 2.0],
    ///     [1.0, 1.0, 3.0, 4.0],
    ///     [2.0, 7.0, 9.0, 0.0],
    ///     [0.0, 0.0, 0.0, 1.0],
    /// ]);
    ///
    /// let elem = matrix.get_mut(1, 1);
    /// elem.map(|el| {
    ///     assert_eq!(*el, 1.0);
    ///     *el = 18.0
    /// });
    ///
    /// assert_eq!(matrix[1][1], 18.0);
    ///
    /// let elem = matrix.get_mut(6, 3);
    /// assert_eq!(elem, None);
    /// ```
    #[must_use]
    #[inline]
    pub const fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        match array_get_mut_checked(&mut self.data, row) {
            Some(row_data) => array_get_mut_checked(row_data.as_mut_slice(), col),
            None => None,
        }
    }

    /// Get a reference to the element at index `row` and `col` without performing any bounds checks.
    ///
    /// # Safety
    ///
    /// You must ensure that `row` and `col` are within the bounds of the `Matrix`,
    /// otherwise this method will cause undefined behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let matrix = Matrix::new([[1]]);
    ///
    /// let data = unsafe { matrix.get_unchecked(0, 0) };
    /// assert_eq!(*data, 1);
    /// ```
    #[must_use]
    #[inline]
    pub const unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T {
        unsafe { array_get_unchecked(array_get_unchecked(&self.data, row).as_slice(), col) }
    }

    /// Get a mutable reference to the element at index `row` and `col` without performing
    /// any bounds checks.
    ///
    /// # Safety
    ///
    /// You must ensure that `row` and `col` are within the bounds of the `Matrix`,
    /// otherwise this method will cause undefined behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let mut matrix = Matrix::new([[1, 2, 3]]);
    ///
    /// let data = unsafe { matrix.get_unchecked_mut(0, 2) };
    /// assert_eq!(*data, 3);
    /// *data = 5;
    ///
    /// assert_eq!(matrix.as_slice(), &[1, 2, 5]);
    /// ```
    #[must_use]
    #[inline]
    pub const unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut T {
        unsafe {
            array_get_unchecked_mut(
                array_get_unchecked_mut(&mut self.data, row).as_mut_slice(),
                col,
            )
        }
    }

    /// Sets the column of the `Matrix` at `col_idx` to the given `col`.
    #[track_caller]
    #[inline]
    pub fn set_col(&mut self, col_idx: usize, col: [T; ROWS]) {
        match self.try_set_col(col_idx, col) {
            Ok(_) => (),
            Err(_) => panic!("Column index out of bounds"),
        }
    }

    #[inline]
    pub fn try_set_col(&mut self, col_idx: usize, col: [T; ROWS]) -> Result<(), [T; ROWS]> {
        if col_idx >= COLS {
            return Err(col);
        }

        let col = ManuallyDrop::new(col);
        for i in 0..COLS {
            unsafe {
                *self.get_unchecked_mut(i, col_idx) = ptr::read(col.get_unchecked(i));
            }
        }

        Ok(())
    }

    #[track_caller]
    #[inline]
    pub fn set_row(&mut self, row_idx: usize, row: [T; COLS]) {
        self.data[row_idx] = row;
    }

    #[inline]
    pub fn try_set_row(&mut self, row_idx: usize, mut row: [T; COLS]) -> Result<(), [T; COLS]> {
        if row_idx >= ROWS {
            return Err(row);
        }

        let matrix_row = unsafe { self.data.get_unchecked_mut(row_idx) };
        mem::swap(&mut row, matrix_row);

        Ok(())
    }

    /// Returns a new `Matrix` where each element is a reference to the corresponding element
    /// in the original matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let matrix = Matrix::new([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    /// ]);
    ///
    /// let matrix_ref = matrix.each_ref();
    ///
    /// for (elem, ref_elem) in matrix.as_slice().iter().zip(matrix_ref.as_slice()) {
    ///     assert_eq!(*elem, **ref_elem);
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub const fn each_ref(&self) -> Matrix<&T, ROWS, COLS> {
        let mut matrix = Matrix::uninit();

        let mut row = 0;
        while row < ROWS {
            let mut col = 0;
            while col < COLS {
                unsafe {
                    matrix
                        .get_unchecked_mut(row, col)
                        .write(self.get_unchecked(row, col));
                }

                col += 1;
            }

            row += 1;
        }

        unsafe { Matrix::assume_init(matrix) }
    }

    /// Returns a new `Matrix` where each element is a mutable reference to the corresponding element
    /// in the original matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use vectral::matrix::Matrix;
    ///
    /// let mut matrix = Matrix::new([
    ///     [1, 2, 3, 4],
    /// ]);
    ///
    /// let mut matrix_mut = matrix.each_mut();
    ///
    /// for (i, elem) in matrix_mut.as_mut_slice().iter_mut().enumerate() {
    ///     **elem = i;
    /// }
    ///
    /// assert_eq!(matrix.as_slice(), [0, 1, 2, 3]);
    /// ```
    #[must_use]
    #[inline]
    pub const fn each_mut(&mut self) -> Matrix<&mut T, ROWS, COLS> {
        let mut matrix = Matrix::uninit();

        let mut row = 0;
        while row < ROWS {
            let mut col = 0;
            while col < COLS {
                unsafe {
                    matrix
                        .get_unchecked_mut(row, col)
                        .write(mem::transmute(&raw mut self.data[row][col]));
                }

                col += 1;
            }

            row += 1;
        }

        unsafe { Matrix::assume_init(matrix) }
    }

    #[inline]
    pub fn zip_map<U, Ret, F: FnMut(T, U) -> Ret>(
        self,
        rhs: Matrix<U, ROWS, COLS>,
        mut f: F,
    ) -> Matrix<Ret, ROWS, COLS> {
        Matrix {
            data: zip_map(self.data, rhs.data, |lhs, rhs| zip_map(lhs, rhs, &mut f)),
        }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn row_ref(&self, n: usize) -> [&T; COLS] {
        match array_get_checked(&self.data, n) {
            Some(row) => row.each_ref(),
            None => panic!("row index out of bounds"),
        }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn row_mut(&mut self, n: usize) -> [&mut T; COLS] {
        match array_get_mut_checked(&mut self.data, n) {
            Some(row) => row.each_mut(),
            None => panic!("row index out of bounds"),
        }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn col_ref(&self, n: usize) -> [&T; ROWS] {
        assert!(n < COLS, "column index out of bounds");

        let mut col = [const { MaybeUninit::uninit() }; ROWS];

        let mut row = 0;
        while row < ROWS {
            unsafe {
                array_get_unchecked_mut(&mut col, row).write(self.get_unchecked(row, n));
            }
            row += 1;
        }

        unsafe { MaybeUninit::assume_init(mem::transmute_copy(&col)) }
    }

    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn col_mut(&mut self, n: usize) -> [&mut T; ROWS] {
        assert!(n < COLS, "column index out of bounds");

        let mut col = [const { MaybeUninit::uninit() }; _];

        let mut row = 0;
        let ptr = self.as_mut_ptr();
        while row < ROWS {
            unsafe {
                array_get_unchecked_mut(&mut col, row).write(&mut *ptr.add(row * ROWS + n));
            }
            row += 1;
        }

        unsafe { MaybeUninit::array_assume_init(col) }
    }

    /// Applies the given function `f` to every element of the `Matrix`, returning
    /// a new matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let mut matrix = Matrix::new([
    ///     [1, 2, 3, 4],
    ///     [5, 6, 7, 8],
    /// ]);
    ///
    /// let transformed: Matrix<String, _, _> = matrix.map(|elem| format!("{elem}"));
    ///
    /// assert_eq!(transformed, Matrix::new([
    ///     [1.to_string(), 2.to_string(), 3.to_string(), 4.to_string()],
    ///     [5.to_string(), 6.to_string(), 7.to_string(), 8.to_string()],
    /// ]));
    /// ```
    #[must_use]
    #[inline]
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Matrix<U, ROWS, COLS> {
        Matrix {
            data: self.data.map(|row| row.map(&mut f)),
        }
    }

    #[must_use]
    #[inline]
    pub fn map_rows<U, F: FnMut([T; COLS]) -> [U; COLS]>(self, f: F) -> Matrix<U, ROWS, COLS> {
        Matrix {
            data: self.data.map(f),
        }
    }

    /// Computes the transpose of the matrix, returning a copy of the transpose.
    ///
    /// This method can be called on matrices of any arbitrary dimension.
    ///
    /// See the [`transpose_in_place()`] method as an alternative for setting a square
    /// matrix to its transpose.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [1, 2, 3, 4],
    ///     [5, 6, 7, 8],
    /// ]);
    ///
    /// assert_eq!(matrix.transpose(), Matrix::new([
    ///     [1, 5],
    ///     [2, 6],
    ///     [3, 7],
    ///     [4, 8],
    /// ]));
    /// ```
    ///
    /// [`transpose_in_place()`]: ./struct.Matrix.html#method.transpose_in_place
    #[must_use]
    #[inline]
    pub const fn transpose(self) -> Matrix<T, COLS, ROWS> {
        let mut transposed = Matrix::uninit();

        let mut row = 0;
        while row < ROWS {
            let mut col = 0;
            while col < COLS {
                unsafe {
                    let write_slot = {
                        let c = array_get_unchecked_mut(&mut transposed.data, col);
                        array_get_unchecked_mut(c, row)
                    };

                    let read_slot = self.get_unchecked(row, col);
                    write_slot.write(ptr::read(read_slot));
                }

                col += 1;
            }
            row += 1;
        }

        let _self = ManuallyDrop::new(self);
        unsafe { Matrix::assume_init(transposed) }
    }

    /// Interprets the data of the matrix as a differently sized matrix, without changing the layout of the data.
    ///
    /// # Example
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let m1 = Matrix::new([[1, 2, 3, 4]]);
    /// let m2 = Matrix::new([
    ///     [1, 2],
    ///     [3, 4],
    /// ]);
    ///
    /// assert_eq!(m1.resize::<2, 2>(), m2);
    /// ```
    ///
    /// A compile time assertion is used to ensure that the number of elements in the matrix cannot grow or shrink:
    ///
    /// ```compile_fail
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([[1, 2, 3, 4]]);
    /// let _ = matrix.resize::<1, 5>();
    /// ```
    ///
    /// ```compile_fail
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([[1, 2, 3, 4]]);
    /// let _ = matrix.resize::<1, 3>();
    /// ```
    #[must_use]
    #[inline]
    pub const fn resize<const NEW_ROWS: usize, const NEW_COLS: usize>(
        self,
    ) -> Matrix<T, NEW_ROWS, NEW_COLS> {
        #[non_exhaustive]
        struct AssertCompatibleMatrixLayout<
            const R0: usize,
            const C0: usize,
            const R1: usize,
            const C1: usize,
        >;

        impl<const R0: usize, const C0: usize, const R1: usize, const C1: usize>
            AssertCompatibleMatrixLayout<R0, C0, R1, C1>
        {
            const ASSERTION: () = assert!(R0 * C0 == R1 * C1);
        }

        #[allow(path_statements)]
        <AssertCompatibleMatrixLayout<ROWS, COLS, NEW_ROWS, NEW_COLS>>::ASSERTION;

        let this = ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&this) }
    }

    /// Returns a new matrix, where every element has been wrapped in a `MaybeUninit`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// # use std::mem::MaybeUninit;
    /// let matrix = Matrix::new([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0],
    /// ]);
    ///
    /// let uninit: Matrix<MaybeUninit<f64>, 2, 2> = matrix.into_uninit();
    /// # let _ = uninit;
    /// ```
    #[must_use]
    #[inline]
    pub const fn into_uninit(self) -> Matrix<MaybeUninit<T>, ROWS, COLS> {
        let this = ManuallyDrop::new(self);
        unsafe { mem::transmute_copy(&this) }
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn into_flattened(self) -> [T; ROWS * COLS] {
        flatten(self.to_array())
    }

    /// Horizontally concatenates two matrices together.
    ///
    /// The two matrices must have the same number of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let mat1 = Matrix::new([
    ///     [01, 02],
    ///     [05, 06],
    ///     [09, 10],
    /// ]);
    ///
    /// let mat2 = Matrix::new([
    ///     [03, 04],
    ///     [07, 08],
    ///     [11, 12],
    /// ]);
    ///
    /// assert_eq!(mat1.concat_horizontal(mat2), Matrix::new([
    ///     [01, 02, 03, 04],
    ///     [05, 06, 07, 08],
    ///     [09, 10, 11, 12],
    /// ]));
    /// ```
    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn concat_horizontal<const NEW_COLS: usize>(
        self,
        matrix: Matrix<T, ROWS, NEW_COLS>,
    ) -> Matrix<T, ROWS, { COLS + NEW_COLS }>
    where
        Matrix<T, ROWS, { COLS + NEW_COLS }>: Sized,
    {
        let mut concat_matrix = Matrix::uninit();

        let mut row = 0;
        while row < ROWS {
            unsafe {
                let src_row = array_get_unchecked(&self.data, row).as_ptr();
                let dst_row = concat_matrix.data.as_mut_ptr().add(row).cast::<T>();

                ptr::copy(src_row, dst_row, COLS);

                let src_row = array_get_unchecked(&matrix.data, row).as_ptr();
                let dst_row = dst_row.add(COLS);

                ptr::copy(src_row, dst_row, NEW_COLS);
            }

            row += 1;
        }

        unsafe {
            let _mats = (ManuallyDrop::new(self), ManuallyDrop::new(matrix));
            Matrix::assume_init(concat_matrix)
        }
    }

    /// Vertically concatenates two matrices together.
    ///
    /// The two matrices must have the same number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let mat1 = Matrix::new([
    ///     [01, 02],
    ///     [03, 04],
    ///     [05, 06],
    /// ]);
    ///
    /// let mat2 = Matrix::new([
    ///     [07, 08],
    ///     [09, 10],
    ///     [11, 12],
    /// ]);
    ///
    /// assert_eq!(mat1.concat_vertical(mat2), Matrix::new([
    ///     [01, 02],
    ///     [03, 04],
    ///     [05, 06],
    ///     [07, 08],
    ///     [09, 10],
    ///     [11, 12],
    /// ]));
    /// ```
    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn concat_vertical<const NEW_ROWS: usize>(
        self,
        matrix: Matrix<T, NEW_ROWS, COLS>,
    ) -> Matrix<T, { ROWS + NEW_ROWS }, COLS>
    where
        Matrix<T, { ROWS + NEW_ROWS }, COLS>: Sized,
    {
        let mut concat_matrix: Matrix<MaybeUninit<T>, _, COLS> = Matrix::uninit();

        let mut row = 0;
        while row < ROWS {
            unsafe {
                let src = { array_get_unchecked(&self.data, row).as_ptr() };
                let dst = {
                    array_get_unchecked_mut(&mut concat_matrix.data, row)
                        .as_mut_ptr()
                        .cast::<T>()
                };
                ptr::copy(src, dst, COLS);
            }

            row += 1;
        }

        while row < ROWS + NEW_ROWS {
            unsafe {
                let idx = row - ROWS;
                let src = array_get_unchecked(&matrix.data, idx).as_ptr();
                let dst = {
                    array_get_unchecked_mut(&mut concat_matrix.data, row)
                        .as_mut_ptr()
                        .cast::<T>()
                };
                ptr::copy(src, dst, COLS);
            }

            row += 1;
        }

        unsafe {
            let _mats = ManuallyDrop::new((self, matrix));
            Matrix::assume_init(concat_matrix)
        }
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub fn into_elems(self) -> array::IntoIter<T, { ROWS * COLS }> {
        IntoIterator::into_iter(self.into_flattened())
    }

    #[must_use]
    #[inline]
    pub fn elems(&self) -> slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    #[must_use]
    #[inline]
    pub fn elems_mut(&mut self) -> slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    #[must_use]
    #[inline]
    pub fn elementwise_add<U>(self, matrix: Matrix<U, ROWS, COLS>) -> Matrix<T::Output, ROWS, COLS>
    where
        T: Add<U>,
    {
        self.zip_map(matrix, Add::add)
    }

    #[must_use]
    #[inline]
    pub fn elementwise_sub<U>(self, matrix: Matrix<U, ROWS, COLS>) -> Matrix<T::Output, ROWS, COLS>
    where
        T: Sub<U>,
    {
        self.zip_map(matrix, Sub::sub)
    }

    #[must_use]
    #[inline]
    pub fn elementwise_mul<U>(self, matrix: Matrix<U, ROWS, COLS>) -> Matrix<T::Output, ROWS, COLS>
    where
        T: Mul<U>,
    {
        self.zip_map(matrix, Mul::mul)
    }

    #[must_use]
    #[inline]
    pub fn elementwise_div<U>(self, matrix: Matrix<U, ROWS, COLS>) -> Matrix<T::Output, ROWS, COLS>
    where
        T: Div<U>,
    {
        self.zip_map(matrix, Div::div)
    }

    /// Creates a new cofactor matrix, where the given `removed_row`. and `removed_col`
    /// are shifted towards the edge of the matrix. Elements are shifted towards the top-left,
    /// and the edges will contain the old row and column.
    ///
    /// # Notes
    ///
    /// It is up to the caller to ensure that the elements on the edges are properly destroyed
    /// if a destructor is required to be called for them. The order of the elements on the
    /// edges is unspecified.
    ///
    /// # Panics
    ///
    /// This method will panic if `removed_row` >= `ROWS`, or `removed_col` >= `COLS`.
    #[track_caller]
    #[must_use]
    #[inline]
    const fn cofactor_shifted_uninit(
        self,
        removed_row: usize,
        removed_col: usize,
    ) -> Matrix<MaybeUninit<T>, ROWS, COLS> {
        let mut cofactor_mat = Matrix::uninit();

        assert!(
            removed_row < ROWS,
            "specified row to remove from cofactor matrix out of bounds"
        );
        assert!(
            removed_col < COLS,
            "specified row to remove from cofactor matrix out of bounds"
        );

        let mut row_idx = 0;
        while row_idx < ROWS {
            let mut col_idx = 0;
            while col_idx < COLS {
                if row_idx == removed_row || col_idx == removed_col {
                    col_idx += 1;
                    continue;
                }

                let cof_row_idx = if row_idx > removed_row {
                    row_idx - 1
                } else {
                    row_idx
                };

                let cof_col_idx = if col_idx > removed_col {
                    col_idx - 1
                } else {
                    col_idx
                };

                unsafe {
                    let src = self.get_unchecked(row_idx, col_idx);
                    let init_dst = cofactor_mat.get_unchecked_mut(cof_row_idx, cof_col_idx);

                    init_dst.write(ptr::read(src));
                }

                col_idx += 1;
            }

            row_idx += 1;
        }

        let mut row_idx = 0;
        while row_idx < ROWS {
            unsafe {
                let dst = cofactor_mat.get_unchecked_mut(row_idx, COLS - 1);
                let src = self.get_unchecked(row_idx, COLS - 1);
                dst.write(ptr::read(src));
            }
            row_idx += 1;
        }

        let mut col_idx = 0;
        while col_idx < ROWS {
            unsafe {
                let dst = cofactor_mat.get_unchecked_mut(ROWS - 1, col_idx);
                let src = self.get_unchecked(ROWS - 1, col_idx);
                dst.write(ptr::read(src));
            }
            col_idx += 1;
        }

        let _self = ManuallyDrop::new(self);
        cofactor_mat
    }
}

impl<T, const ROWS: usize, const COLS: usize> Matrix<MaybeUninit<T>, ROWS, COLS> {
    #[must_use]
    #[inline]
    pub const unsafe fn assume_init(self) -> Matrix<T, ROWS, COLS> {
        unsafe { mem::transmute_copy(&self) }
    }
}

impl<T, const COLS: usize> Matrix<T, 1, COLS> {
    /// Creates a new single-row matrix from the given vector.
    #[must_use]
    #[inline]
    pub const fn from_row_vector(vector: Vector<T, COLS>) -> Self {
        Self::new([vector.to_array()])
    }

    /// Converts the matrix into a row vector of `COLS` elements.
    #[must_use]
    #[inline]
    pub const fn into_row_vector(self) -> Vector<T, COLS> {
        let Matrix { data: [ref item] } = self;
        let array = unsafe { ptr::read(item) };

        let _ = ManuallyDrop::new(self);

        Vector::new(array)
    }
}

impl<T, const ROWS: usize> Matrix<T, ROWS, 1> {
    /// Creates a new single-column matrix from the given vector.
    #[must_use]
    #[inline]
    pub const fn from_column_vector(vector: Vector<T, ROWS>) -> Self {
        Matrix::from_row_vector(vector).transpose()
    }

    /// Converts the matrix into a column vector of `ROWS` elements.
    #[must_use]
    #[inline]
    pub const fn into_column_vector(self) -> Vector<T, ROWS> {
        Matrix::into_row_vector(self.transpose())
    }
}

impl<T: Copy, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Creates a new matrix, where every element of `Matrix` is set to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix: Matrix<_, 4, 4> = Matrix::splat(21);
    /// assert!(matrix.as_slice().into_iter().all(|elem| *elem == 21));
    /// ```
    #[must_use]
    #[inline]
    pub const fn splat(value: T) -> Self {
        Self {
            data: [[value; COLS]; ROWS],
        }
    }

    /// Returns a copy of the column at `n`.
    ///
    /// # Panics
    ///
    /// This method will panic if `n` is equal or greater to `COLS`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [0, 1, 2, 3, 4],
    ///     [5, 6, 7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(matrix.col(1), [1, 6]);
    /// assert_eq!(matrix.col(4), [4, 9]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn col(&self, n: usize) -> [T; ROWS] {
        assert!(n < COLS, "given column index is out of bounds");

        // `MaybeUninit` is used here because:
        // (a). using `self[0][0]` would panic if `ROWS == 0 && COLS == 0`.
        // (b). it avoids requiring a trait-specified default (`T: Zero` or `T: Default``).
        let mut col = [MaybeUninit::uninit(); ROWS];

        let mut row = 0;
        while row < ROWS {
            unsafe {
                col[row].write(ptr::read(self.get_unchecked(row, n)));
            }
            row += 1;
        }

        unsafe { MaybeUninit::array_assume_init(col) }
    }

    /// Returns a copy of the row at `n`.
    ///
    /// # Panics
    ///
    /// This method will panic if `n` is equal or greater to `ROWS`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [0, 1, 2, 3, 4],
    ///     [5, 6, 7, 8, 9]
    /// ]);
    ///
    /// assert_eq!(matrix.row(1), [5, 6, 7, 8, 9]);
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn row(&self, n: usize) -> [T; COLS] {
        assert!(n < ROWS, "given row index is out of bounds");
        let mut row = [MaybeUninit::uninit(); COLS];

        unsafe {
            ptr::copy(
                self.data.as_ptr().add(n).cast::<T>(),
                row.as_mut_ptr().cast(),
                COLS,
            );
        }

        unsafe { MaybeUninit::array_assume_init(row) }
    }

    #[must_use]
    #[inline]
    pub const fn try_swizzle<const ROWS2: usize, const COLS2: usize>(
        &self,
        swizzle_matrix: &[[(usize, usize); COLS2]; ROWS2],
    ) -> Option<Matrix<T, ROWS2, COLS2>> {
        let mut matrix = Matrix::uninit();

        let mut row = 0;
        while row < ROWS2 {
            let mut col = 0;
            while col < COLS2 {
                let (swizzle_row_idx, swizzle_col_idx) = unsafe {
                    let swizz_row = array_get_unchecked(swizzle_matrix, row);
                    *array_get_unchecked(swizz_row, col)
                };

                if swizzle_row_idx >= ROWS || swizzle_col_idx >= COLS {
                    return None;
                }

                unsafe {
                    let elem = *self.get_unchecked(swizzle_row_idx, swizzle_col_idx);

                    let slot = matrix.get_unchecked_mut(row, col);
                    slot.write(elem);
                }
                col += 1;
            }
            row += 1;
        }

        unsafe { Some(Matrix::assume_init(matrix)) }
    }

    #[must_use]
    #[inline]
    pub const fn swizzle_or<const SWIZZ_ROWS: usize, const SWIZZ_COLS: usize>(
        self,
        swizzle_matrix: &[[(usize, usize); SWIZZ_COLS]; SWIZZ_ROWS],
        or: &Matrix<T, SWIZZ_ROWS, SWIZZ_COLS>,
    ) -> Matrix<T, SWIZZ_ROWS, SWIZZ_COLS> {
        let mut matrix = Matrix::uninit();

        let mut row = 0;
        while row < SWIZZ_ROWS {
            let mut col = 0;
            while col < SWIZZ_COLS {
                let (swizzle_row_idx, swizzle_col_idx) = unsafe {
                    let swizz_row = array_get_unchecked(swizzle_matrix, row);
                    *array_get_unchecked(swizz_row, col)
                };

                unsafe {
                    let elem = if swizzle_row_idx >= ROWS || swizzle_col_idx >= COLS {
                        *or.get_unchecked(swizzle_row_idx, swizzle_col_idx)
                    } else {
                        *self.get_unchecked(swizzle_row_idx, swizzle_col_idx)
                    };

                    let slot = matrix.get_unchecked_mut(row, col);
                    slot.write(elem);
                }
                col += 1;
            }
            row += 1;
        }

        unsafe { Matrix::assume_init(matrix) }
    }

    /// Swizzle the matrix using the given swizzle matrix.
    ///
    /// The returned matrix will have the dimensions of the swizzle matrix, where each
    /// element will be the element found at the coordinate of the original matrix.
    ///
    /// # Panics
    ///
    /// If any coordinate member is out of bounds from `self`, then this method
    /// will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [1, 2, 3, 4],
    ///     [5, 6, 7, 8],
    /// ]);
    ///
    /// let swizzled = matrix.swizzle(&[
    ///     [(0, 0), (0, 3)],
    ///     [(1, 0), (1, 1)],
    /// ]);
    ///
    /// assert_eq!(swizzled, Matrix::new([
    ///     [1, 4],
    ///     [5, 6],
    /// ]));
    /// ```
    #[must_use]
    #[inline]
    pub const fn swizzle<const SWIZZ_ROWS: usize, const SWIZZ_COLS: usize>(
        &self,
        swizzle_matrix: &[[(usize, usize); SWIZZ_COLS]; SWIZZ_ROWS],
    ) -> Matrix<T, SWIZZ_ROWS, SWIZZ_COLS> {
        match self.try_swizzle(swizzle_matrix) {
            Some(matrix) => matrix,
            None => panic!("swizzle index out of bounds"),
        }
    }
}

impl<T, const N: usize> Matrix<T, N, N> {
    /// Returns a slice of references to the rightwards diagonal of the matrix (starting at the
    /// top-left, and going towards the bottom-right).
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [1, 0, 0],
    ///     [0, 2, 0],
    ///     [0, 0, 3],
    /// ]);
    ///
    /// let diagonal = matrix.rightwards_diagonal_ref();
    /// let expected_diagonal = [1, 2, 3];
    /// for (matrix_elem, test) in diagonal.iter().zip(expected_diagonal.into_iter()) {
    ///     assert_eq!(**matrix_elem, test);
    /// }
    /// ```
    #[must_use]
    #[inline]
    pub const fn rightwards_diagonal_ref(&self) -> [&T; N] {
        let mut diagonal = [const { MaybeUninit::uninit() }; _];
        let mut i = 0;
        while i < N {
            unsafe {
                array_get_unchecked_mut(&mut diagonal, i).write(self.get_unchecked(i, i));
            }
            i += 1;
        }

        unsafe { MaybeUninit::array_assume_init(diagonal) }
    }

    /// Returns a slice of references to the rightwards diagonal of the matrix (starting at the
    /// top-right, and going towards the bottom-left).
    #[must_use]
    #[inline]
    pub const fn leftwards_diagonal_ref(&self) -> [&T; N] {
        let mut diagonal = [const { MaybeUninit::uninit() }; _];
        let (mut row, mut col) = (0, N - 1);
        while row < N {
            unsafe {
                array_get_unchecked_mut(&mut diagonal, row).write(self.get_unchecked(row, col));
            }
            row += 1;
            col = col.saturating_sub(1);
        }

        unsafe { MaybeUninit::array_assume_init(diagonal) }
    }

    /// Returns a slice of mutable references to the rightwards diagonal of the matrix (starting at the
    /// top-left, and going towards the bottom right).
    #[must_use]
    #[inline]
    pub const fn rightwards_diagonal_mut(&mut self) -> [&mut T; N] {
        let mut diagonal = [const { MaybeUninit::uninit() }; _];
        let mut i = 0;
        let ptr = self.as_mut_ptr();
        while i < N {
            unsafe {
                array_get_unchecked_mut(&mut diagonal, i).write(&mut *ptr.add(i * N + i));
            }
            i += 1;
        }

        unsafe { MaybeUninit::array_assume_init(diagonal) }
    }

    #[must_use]
    #[inline]
    pub const fn leftwards_diagonal_mut(&mut self) -> [&mut T; N] {
        let mut diagonal = [const { MaybeUninit::uninit() }; _];
        let (mut row, mut col) = (0, N - 1);
        let ptr = self.as_mut_ptr();
        while row < N {
            unsafe {
                array_get_unchecked_mut(&mut diagonal, row).write(&mut *ptr.add(row * N + col));
            }
            row += 1;
            col = col.saturating_sub(1);
        }

        unsafe { MaybeUninit::array_assume_init(diagonal) }
    }

    /// Sets the rightwards diagonal of the matrix (starting at the top left) to the given
    /// `new_diagonal`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let mut identity_matrix = Matrix::<f32, 4, 4>::splat(0.0);
    /// identity_matrix.set_rightwards_diagonal([1.0; 4]);
    /// assert_eq!(identity_matrix, Matrix::identity());
    /// ```
    #[inline]
    pub fn set_rightwards_diagonal(&mut self, new_diagonal: [T; N]) {
        let mut i = 0;
        while i < N {
            unsafe {
                *self.get_unchecked_mut(i, i) = ptr::read(new_diagonal.get_unchecked(i));
            }
            i += 1;
        }

        let _diag = ManuallyDrop::new(new_diagonal);
    }

    #[inline]
    pub fn set_leftwards_diagonal(&mut self, new_diagonal: [T; N]) {
        let (mut row, mut col) = (0, N - 1);
        while row < N {
            unsafe {
                *self.get_unchecked_mut(row, col) = ptr::read(new_diagonal.get_unchecked(row));
            }
            row += 1;
            col = col.saturating_sub(1);
        }

        let _diag = ManuallyDrop::new(new_diagonal);
    }

    /// Transposes the matrix by swapping elements around the diagonal, without creating
    /// a new intermediate matrix.
    ///
    /// To compute the transpose of an arbitrarily-sized matrix, use the [`transpose()`] method.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let mut matrix = Matrix::new([
    ///     [1, 2, 3],
    ///     [4, 5, 6],
    ///     [7, 8, 9],
    /// ]);
    ///
    /// matrix.transpose_in_place();
    ///
    /// let transposed = Matrix::new([
    ///     [1, 4, 7],
    ///     [2, 5, 8],
    ///     [3, 6, 9],
    /// ]);
    ///
    /// assert_eq!(matrix, transposed);
    /// ```
    ///
    /// [`transpose()`]: ./struct.Matrix.html#method.transpose
    #[inline]
    pub const fn transpose_in_place(&mut self) {
        let slice: &mut [T] = self.as_mut_slice();
        let mut row = 0;
        while row < N {
            let mut col = row + 1;
            while col < N {
                slice.swap(row * N + col, col * N + row);
                col += 1;
            }
            row += 1;
        }
    }
}

impl<T: Copy, const N: usize> Matrix<T, N, N> {
    #[must_use]
    #[inline]
    pub const fn rightwards_diagonal(&self) -> [T; N] {
        let mut diagonal = [const { MaybeUninit::uninit() }; _];
        let mut i = 0;
        while i < N {
            unsafe {
                array_get_unchecked_mut(&mut diagonal, i)
                    .write(ptr::read(self.get_unchecked(i, i)));
            }
            i += 1;
        }

        unsafe { MaybeUninit::array_assume_init(diagonal) }
    }

    #[must_use]
    #[inline]
    pub const fn leftwards_diagonal(&self) -> [T; N] {
        let mut diagonal = [const { MaybeUninit::uninit() }; _];
        let (mut row, mut col) = (0, N - 1);
        while row < N {
            unsafe {
                array_get_unchecked_mut(&mut diagonal, row)
                    .write(ptr::read(self.get_unchecked(row, col)));
            }
            row += 1;
            col = col.saturating_sub(1);
        }

        unsafe { MaybeUninit::array_assume_init(diagonal) }
    }

    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub const fn extract_translation(self) -> Vector<T, { N - 1 }> {
        let col = self.col(N - 1);
        Vector::new(shrink_to_copy(col))
    }
}

impl<T, const ROWS: usize, const COLS: usize> Index<usize> for Matrix<T, ROWS, COLS> {
    type Output = [T; COLS];
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const ROWS: usize, const COLS: usize> IndexMut<usize> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const ROWS: usize, const COLS: usize> Index<(usize, usize)> for Matrix<T, ROWS, COLS> {
    type Output = T;
    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row][col]
    }
}

impl<T, const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row][col]
    }
}

impl<T: Copy, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Returns the cofactor of this matrix.
    ///
    /// The cofactor matrix is the matrix calculated by removing the given `removed_row` and
    /// `removed_col` from the matrix.
    ///
    /// # Panics
    ///
    /// This method will panic if either `removed_row` >= `ROWS`, or `removed_col` >= `COLS`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [01, 02, 03, 04],
    ///     [05, 06, 07, 08],
    ///     [09, 10, 11, 12],
    ///     [13, 14, 15, 16],
    /// ]);
    ///
    /// let cofactor = matrix.cofactor(1, 2);
    ///
    /// assert_eq!(cofactor, Matrix::new([
    ///     [01, 02, 04],
    ///     [09, 10, 12],
    ///     [13, 14, 16],
    /// ]));
    /// ```
    #[cfg(feature = "nightly")]
    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn cofactor(
        self,
        removed_row: usize,
        removed_col: usize,
    ) -> Matrix<T, { ROWS - 1 }, { COLS - 1 }>
    where
        Matrix<T, { ROWS - 1 }, { COLS - 1 }>: Sized,
    {
        let mut return_matrix = Matrix::splat(const { MaybeUninit::uninit() });
        let cofactor_matrix = Matrix::cofactor_shifted_uninit(self, removed_row, removed_col);

        let mut row = 0;
        while row < ROWS - 1 {
            let cofactor_row_start = row * ROWS;
            let return_matrix_start = row * (ROWS - 1);

            unsafe {
                ptr::copy_nonoverlapping(
                    cofactor_matrix.as_ptr().add(cofactor_row_start),
                    return_matrix.as_mut_ptr().add(return_matrix_start),
                    COLS - 1,
                );
            }

            row += 1;
        }

        unsafe { Matrix::assume_init(return_matrix) }
    }
}

impl<T: Copy, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn cofactor_shifted_with(
        self,
        value: T,
        removed_row: usize,
        removed_col: usize,
    ) -> Self {
        let mut return_matrix = Matrix::cofactor_shifted_uninit(self, removed_row, removed_col);

        let mut row = 0;
        while row < ROWS {
            unsafe {
                return_matrix.get_unchecked_mut(row, COLS - 1).write(value);
            }
            row += 1;
        }

        let mut col = 0;
        while col < COLS {
            unsafe {
                return_matrix.get_unchecked_mut(ROWS - 1, col).write(value);
            }
            col += 1;
        }

        unsafe { Matrix::assume_init(return_matrix) }
    }
}

impl<T: Copy + Zero, const ROWS: usize, const COLS: usize> Matrix<T, ROWS, COLS> {
    /// Returns the cofactor of this matrix, where the retained elements are shifted to
    /// the top-left of the matrix, and the edges are zero-filled.
    ///
    /// The cofactor matrix is the matrix calculated by removing the given `removed_row` and
    /// `removed_col` from the matrix.
    ///
    /// # Panics
    ///
    /// This method will panic if either `removed_row` >= `ROWS`, or `removed_col` >= `COLS`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::new([
    ///     [01, 02, 03, 04],
    ///     [05, 06, 07, 08],
    ///     [09, 10, 11, 12],
    ///     [13, 14, 15, 16],
    /// ]);
    ///
    /// let cofactor = matrix.cofactor_shifted(1, 2);
    ///
    /// assert_eq!(cofactor, Matrix::new([
    ///     [01, 02, 04, 00],
    ///     [09, 10, 12, 00],
    ///     [13, 14, 16, 00],
    ///     [00, 00, 00, 00],
    /// ]));
    /// ```
    #[track_caller]
    #[must_use]
    #[inline]
    pub const fn cofactor_shifted(self, removed_row: usize, removed_col: usize) -> Self {
        self.cofactor_shifted_with(Zero::ZERO, removed_row, removed_col)
    }
}

impl<T: Zero, const ROWS: usize, const COLS: usize> Zero for Matrix<T, ROWS, COLS> {
    const ZERO: Self = Matrix::new(Zero::ZERO);
}

impl<T: Zero + One, const N: usize> Matrix<T, N, N> {
    /// Constructs an instance of the identity matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vectral::matrix::Matrix;
    /// let matrix = Matrix::<f64, 4, 4>::identity();
    ///
    /// assert_eq!(matrix, Matrix::new([
    ///     [1.0, 0.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0, 0.0],
    ///     [0.0, 0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 0.0, 1.0],
    /// ]));
    /// ```
    #[must_use]
    #[inline]
    pub const fn identity() -> Self {
        let mut mat = Matrix::ZERO.into_uninit();

        let mut i = 0;
        while i < N {
            unsafe {
                mat.get_unchecked_mut(i, i).write(T::ONE);
            }
            i += 1;
        }

        unsafe { Matrix::assume_init(mat) }
    }
}

impl<T, const A: usize, const B: usize, const C: usize> Mul<Matrix<T, B, C>> for Matrix<T, A, B>
where
    T: Zero + Copy + ClosedMul + ClosedAdd,
{
    type Output = Matrix<T, A, C>;
    #[inline]
    fn mul(self, rhs: Matrix<T, B, C>) -> Self::Output {
        let mut out_matrix: Matrix<_, A, C> = Matrix::uninit();

        for row in 0..A {
            for col in 0..C {
                let x = self.row(row);
                let y = rhs.col(col);

                unsafe {
                    out_matrix
                        .get_unchecked_mut(row, col)
                        .write(sum(zip_map(x, y, Mul::mul)));
                }
            }
        }

        unsafe { Matrix::assume_init(out_matrix) }
    }
}

impl<T, const ROWS: usize, const COLS: usize> From<[[T; COLS]; ROWS]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn from(value: [[T; COLS]; ROWS]) -> Self {
        Self::new(value)
    }
}

impl<T, const ROWS: usize, const COLS: usize> From<Matrix<T, ROWS, COLS>> for [[T; COLS]; ROWS] {
    #[inline]
    fn from(value: Matrix<T, ROWS, COLS>) -> Self {
        value.data
    }
}

impl<T: Copy + ClosedAdd + ClosedMul + ClosedSub + One + Zero> From<Quaternion<T>> for Matrix4<T> {
    #[inline]
    fn from(value: Quaternion<T>) -> Self {
        Matrix::rotation_3d(value)
    }
}

impl<T: Copy + Mul, const ROWS: usize, const COLS: usize> Mul<T> for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T::Output, ROWS, COLS>;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Matrix {
            data: self.data.map(|row| row.map(|item| item * rhs)),
        }
    }
}

impl<T: MulAssign<U>, U: Copy, const ROWS: usize, const COLS: usize> MulAssign<U>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn mul_assign(&mut self, rhs: U) {
        for elem in self.as_mut_slice() {
            elem.mul_assign(rhs);
        }
    }
}

impl<T: DivAssign<U>, U: Copy, const ROWS: usize, const COLS: usize> DivAssign<U>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn div_assign(&mut self, rhs: U) {
        for elem in self.as_mut_slice() {
            elem.div_assign(rhs);
        }
    }
}

impl<T: Add<U>, U: Copy, const ROWS: usize, const COLS: usize> Add<Matrix<U, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T::Output, ROWS, COLS>;
    #[inline]
    fn add(self, rhs: Matrix<U, ROWS, COLS>) -> Self::Output {
        self.zip_map(rhs, Add::add)
    }
}

impl<T: AddAssign<U>, U, const ROWS: usize, const COLS: usize> AddAssign<Matrix<U, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn add_assign(&mut self, rhs: Matrix<U, ROWS, COLS>) {
        self.each_mut().zip_map(rhs, |x, y| x.add_assign(y));
    }
}

impl<T: Sub<U>, U, const ROWS: usize, const COLS: usize> Sub<Matrix<U, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    type Output = Matrix<T::Output, ROWS, COLS>;
    #[inline]
    fn sub(self, rhs: Matrix<U, ROWS, COLS>) -> Self::Output {
        self.zip_map(rhs, Sub::sub)
    }
}

impl<T: SubAssign<U>, U, const ROWS: usize, const COLS: usize> SubAssign<Matrix<U, ROWS, COLS>>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Matrix<U, ROWS, COLS>) {
        self.each_mut().zip_map(rhs, |x, y| x.sub_assign(y));
    }
}

impl<T: Neg, const ROWS: usize, const COLS: usize> Neg for Matrix<T, ROWS, COLS> {
    type Output = Matrix<T::Output, ROWS, COLS>;

    #[inline]
    fn neg(self) -> Self::Output {
        self.map(|elem| elem.neg())
    }
}

#[cfg(feature = "simd")]
impl<T, const A: usize, const B: usize, const C: usize> SimdMul<Matrix<T, B, C>> for Matrix<T, A, B>
where
    T: Zero + ClosedAdd + ClosedMul + SimdElement,
    Simd<T, B>: ClosedMul,
    LaneCount<B>: SupportedLaneCount,
{
    type Output = Matrix<T, A, C>;
    #[inline]
    fn simd_mul(self, rhs: Matrix<T, B, C>) -> Self::Output {
        let mut out_matrix: Matrix<_, A, C> = Matrix::uninit();

        for row in 0..A {
            for col in 0..C {
                let x = Simd::from_array(self.row(row));
                let y = Simd::from_array(rhs.col(col));

                unsafe {
                    out_matrix
                        .get_unchecked_mut(row, col)
                        .write(sum((x * y).to_array()));
                }
            }
        }

        unsafe { Matrix::assume_init(out_matrix) }
    }
}

#[cfg(feature = "simd")]
impl<T, const ROWS: usize, const COLS: usize> SimdMul<Vector<T, COLS>> for Matrix<T, ROWS, COLS>
where
    T: Zero + ClosedAdd + ClosedMul + SimdElement,
    Simd<T, COLS>: ClosedMul,
    LaneCount<COLS>: SupportedLaneCount,
{
    type Output = Vector<T, ROWS>;
    #[inline]
    fn simd_mul(self, rhs: Vector<T, COLS>) -> Self::Output {
        let rhs = Simd::from_array(rhs.to_array());

        let result = self.data.map(|row| {
            let row = Simd::from_array(row);
            sum((row * rhs).to_array())
        });

        Vector::from(result)
    }
}

#[cfg(feature = "simd")]
impl<T, const ROWS: usize, const COLS: usize> SimdMul<T> for Matrix<T, ROWS, COLS>
where
    T: Zero + ClosedAdd + ClosedMul + SimdElement,
    Simd<T, { ROWS * COLS }>: ClosedMul,
    LaneCount<{ ROWS * COLS }>: SupportedLaneCount,
{
    type Output = Matrix<T, ROWS, COLS>;
    #[inline]
    fn simd_mul(self, rhs: T) -> Self::Output {
        let simd = Simd::splat(rhs);
        let this = Simd::from_array(self.into_flattened());
        let result = this * simd;
        Matrix::new([result.to_array()]).resize::<ROWS, COLS>()
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: AddAssign + SubAssign + Zero + Copy + ClosedAdd + ClosedMul + ClosedSub,
{
    /// Calculates the scalar determinant of the matrix.
    ///
    /// The determinant is a value used to calculate the inverse of the matrix. It is calculated
    /// by recursively accumulating the determinants of cofactor matrices.
    ///
    /// If this value is `0`, then the matrix is not invertible.
    #[must_use]
    #[inline]
    pub fn determinant(self) -> T {
        self.det_inner(N)
    }

    #[must_use]
    #[inline]
    fn det_inner(self, n: usize) -> T {
        match n {
            0 => Zero::ZERO,
            1 => self[0][0],
            2 => (self[0][0] * self[1][1]) - (self[0][1] * self[1][0]),
            _ => {
                let mut result = Zero::ZERO;
                let top_row = self.row(0);

                let mut should_sub = false;
                for (col_idx, col_elem) in top_row.iter().enumerate() {
                    let cofactor = self.cofactor_shifted(0, col_idx);

                    let value = *col_elem * cofactor.det_inner(n - 1);
                    if should_sub {
                        result -= value;
                    } else {
                        result += value;
                    }

                    should_sub = !should_sub;
                }

                result
            }
        }
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: AddAssign + SubAssign + Zero + Copy + ClosedAdd + ClosedMul + ClosedSub + PartialEq,
{
    /// Returns whether the matrix is invertible.
    ///
    /// If this is `false`, than [`Matrix::inverse()`] will fail.
    ///
    /// [`Matrix::inverse()`]: ./struct.Matrix.html#method.inverse
    #[must_use]
    #[inline]
    pub fn has_inverse(self) -> bool {
        self.determinant() != Zero::ZERO
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: AddAssign + ClosedAdd + Copy + ClosedMul + ClosedSub + SubAssign + One + Zero + ClosedNeg,
{
    /// Calculates the adjoint matrix of the given matrix.
    ///
    /// This is used to find the inverse of the matrix in [`Matrix::inverse()`].
    ///
    /// [`Matrix::inverse()`]: ./struct.Matrix.html#method.inverse
    #[doc(alias = "adjugate")]
    #[must_use]
    #[inline]
    pub fn adjoint(self) -> Matrix<T, N, N> {
        if N <= 1 {
            return Matrix::splat(T::ONE);
        }

        let mut adjoint_mat = Matrix::splat(MaybeUninit::uninit());

        let mut is_neg = false;
        for row in 0..N {
            for col in 0..N {
                let cofactor_matrix: Matrix<T, N, N> = self.cofactor_shifted(row, col);
                let cofactor_scalar = cofactor_matrix.det_inner(N - 1);
                let adjoint_slot = &mut adjoint_mat[row][col];

                if is_neg {
                    adjoint_slot.write(cofactor_scalar.neg());
                } else {
                    adjoint_slot.write(cofactor_scalar);
                }

                is_neg = !is_neg;
            }

            if N.is_multiple_of(2) {
                is_neg = !is_neg;
            }
        }

        adjoint_mat.transpose_in_place();
        unsafe { Matrix::assume_init(adjoint_mat) }
    }
}

impl<T, const ROWS: usize, const COLS: usize> AsRef<[T]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const ROWS: usize, const COLS: usize> AsMut<[T]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const ROWS: usize, const COLS: usize> Borrow<[T]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const ROWS: usize, const COLS: usize> BorrowMut<[T]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const ROWS: usize, const COLS: usize> AsRef<[[T; COLS]; ROWS]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn as_ref(&self) -> &[[T; COLS]; ROWS] {
        self.as_array()
    }
}

impl<T, const ROWS: usize, const COLS: usize> AsMut<[[T; COLS]; ROWS]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn as_mut(&mut self) -> &mut [[T; COLS]; ROWS] {
        self.as_array_mut()
    }
}

impl<T, const ROWS: usize, const COLS: usize> Borrow<[[T; COLS]; ROWS]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn borrow(&self) -> &[[T; COLS]; ROWS] {
        self.as_array()
    }
}

impl<T, const ROWS: usize, const COLS: usize> BorrowMut<[[T; COLS]; ROWS]>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn borrow_mut(&mut self) -> &mut [[T; COLS]; ROWS] {
        self.as_array_mut()
    }
}

#[cfg(feature = "nightly")]
impl<T, const ROWS: usize, const COLS: usize> AsRef<[T; ROWS * COLS]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn as_ref(&self) -> &[T; ROWS * COLS] {
        self.as_flattened_array()
    }
}

#[cfg(feature = "nightly")]
impl<T, const ROWS: usize, const COLS: usize> AsMut<[T; ROWS * COLS]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; ROWS * COLS] {
        self.as_mut_flattened_array()
    }
}

#[cfg(feature = "nightly")]
impl<T, const ROWS: usize, const COLS: usize> Borrow<[T; ROWS * COLS]> for Matrix<T, ROWS, COLS> {
    #[inline]
    fn borrow(&self) -> &[T; ROWS * COLS] {
        self.as_flattened_array()
    }
}

#[cfg(feature = "nightly")]
impl<T, const ROWS: usize, const COLS: usize> BorrowMut<[T; ROWS * COLS]>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T; ROWS * COLS] {
        self.as_mut_flattened_array()
    }
}

/// A 4x4 matrix.
pub type Matrix4<T = f32> = Matrix<T, 4, 4>;

/// A 3x3 matrix.
pub type Matrix3<T = f32> = Matrix<T, 3, 3>;

#[cfg(feature = "nightly")]
impl<T: One + Zero, const N: usize> Matrix<T, N, N>
where
    Vector<T, { N - 1 }>: Sized,
{
    /// Create a new translation matrix from the given `offset`, used for transforming
    /// the positions of vectors in homogeneous space.
    ///
    /// # Example
    ///
    /// ```
    /// # use vectral::{matrix::Matrix4, vector::Vector3};
    /// let offset = Vector3::new([1.0, 2.0, 3.0]);
    /// let transform_matrix = Matrix4::translation(offset);
    ///
    /// assert_eq!(transform_matrix.col(3), [1.0, 2.0, 3.0, 1.0]);
    /// ```
    #[cfg(feature = "nightly")]
    #[must_use]
    #[inline]
    pub fn translation(offset: Vector<T, { N - 1 }>) -> Self {
        let mut mat = Matrix::identity();
        let col = mat.col_mut(N - 1);

        for (mat_elem, v_elem) in col.into_iter().zip(offset.to_array()) {
            *mat_elem = v_elem;
        }

        mat
    }

    /// Create a new scaling matrix from the given `scale`, used for transforming
    /// the scale of vectors in homogeneous space.
    ///
    /// # Example
    ///
    /// ```
    /// # use vectral::{matrix::Matrix3, vector::Vector2};
    /// let scale = Vector2::splat(5.0);
    /// let scale_matrix = Matrix3::scaling(scale);
    ///
    /// assert_eq!(scale_matrix.rightwards_diagonal(), [5.0, 5.0, 1.0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn scaling(scale: Vector<T, { N - 1 }>) -> Self {
        let mut mat = Matrix::identity();
        let diag = mat.rightwards_diagonal_mut();

        for (mat_elem, v_elem) in diag.into_iter().zip(scale.to_array()) {
            *mat_elem = v_elem;
        }

        mat
    }
}

impl<T: Zero + One + Trig + Copy + ClosedNeg> Matrix4<T> {
    /// Returns a 3D rotation matrix which will transform objects by `angle` around
    /// the X-axis.
    #[must_use]
    #[inline]
    pub fn x_axis_rotation(angle: Angle<T>) -> Self {
        let (sin_t, cos_t) = T::sin_cos(angle.in_radians());

        let mut result = Matrix::identity();
        result[1][1] = cos_t;
        result[1][2] = sin_t;
        result[2][1] = -sin_t;
        result[2][2] = cos_t;

        result
    }

    /// Returns a 3D rotation matrix which will transform objects by `angle` around
    /// the Y-axis.
    #[must_use]
    #[inline]
    pub fn y_axis_rotation(angle: Angle<T>) -> Self {
        let (sin_t, cos_t) = T::sin_cos(angle.in_radians());

        let mut result = Matrix::identity();
        result[0][0] = cos_t;
        result[0][2] = -sin_t;
        result[2][0] = sin_t;
        result[2][2] = cos_t;

        result
    }

    /// Returns a 3D rotation matrix which will transform objects by `angle` around
    /// the Z-axis.
    #[must_use]
    #[inline]
    pub fn z_axis_rotation(angle: Angle<T>) -> Self {
        let (sin_t, cos_t) = T::sin_cos(angle.in_radians());

        let mut result = Matrix::identity();
        result[0][0] = cos_t;
        result[0][1] = sin_t;
        result[1][0] = -sin_t;
        result[1][1] = cos_t;

        result
    }
}

impl<T: Trig + One + ClosedDiv + ClosedAdd + ClosedMul + ClosedSub + Zero> Matrix4<T> {
    /// Returns a 3D rotation matrix which will transform objects by `angle` around
    /// the given `axis` vector.
    #[must_use]
    #[inline]
    pub fn axis_rotation_3d(angle: Angle<T>, axis: Vector3<T>) -> Self {
        let quat = Quaternion::from_angle_axis(angle, axis);
        Matrix::rotation_3d(quat)
    }
}

impl<T: Copy + ClosedAdd + ClosedMul + ClosedSub + One + Zero> Matrix4<T> {
    #[must_use]
    #[inline]
    pub fn rotation_3d(q: Quaternion<T>) -> Self {
        let one = T::ONE;
        let two = T::ONE + T::ONE;

        #[inline(always)]
        #[must_use]
        fn sq<T: Copy + Mul>(val: T) -> T::Output {
            val * val
        }

        let mut matrix = Matrix::identity();

        let col_1 = [
            one - (two * (sq(q.v.y) + sq(q.v.z))),
            two * ((q.v.x * q.v.y) - (q.v.z * q.w)),
            two * ((q.v.x * q.v.z) + (q.v.y * q.w)),
            Zero::ZERO,
        ];

        let col_2 = [
            two * ((q.v.x * q.v.y) + (q.v.z * q.w)),
            one - (two * (sq(q.v.x) + sq(q.v.z))),
            two * ((q.v.y * q.v.z) - (q.v.x * q.w)),
            Zero::ZERO,
        ];

        let col_3 = [
            two * ((q.v.x * q.v.z) - (q.v.y * q.w)),
            two * ((q.v.y * q.v.z) + (q.v.x * q.w)),
            one - (two * (sq(q.v.x) + sq(q.v.y))),
            Zero::ZERO,
        ];

        matrix.set_col(0, col_1);
        matrix.set_col(1, col_2);
        matrix.set_col(2, col_3);

        matrix
    }
}

#[cfg(feature = "nightly")]
impl<T> Matrix4<T>
where
    T: ClosedAdd + ClosedMul + CheckedDiv<Output = T> + ClosedSub + Sqrt + One + Zero + ClosedNeg,
{
    #[must_use]
    #[inline]
    pub fn look_at_lh(origin: Point3<T>, target: Point3<T>, up: Vector3<T>) -> Self {
        let mut cam_to_world = Matrix4::translation(origin.into());

        let dir = origin.direction_to(target);
        let left = Vector::cross(up.normalized(), dir);
        let new_up = Vector::cross(dir, left);

        cam_to_world.set_col(0, left.expand(T::ZERO).to_array());
        cam_to_world.set_col(1, new_up.expand(T::ZERO).to_array());
        cam_to_world.set_col(2, dir.expand(T::ZERO).to_array());

        cam_to_world
    }
}

impl<T> Matrix<T, 4, 4>
where
    T: ClosedAdd + ClosedDiv + ClosedSub + ClosedNeg + Copy + Zero + One,
{
    #[must_use]
    #[inline]
    pub fn orthographic_projection_3d(
        left: T,
        right: T,
        bottom: T,
        top: T,
        z_near: T,
        z_far: T,
    ) -> Self {
        let mut ortho = Matrix::identity();
        let two = T::ONE + T::ONE;

        let diag = [
            two / (right - left),
            two / (top - bottom),
            -two / (z_near - z_far),
            One::ONE,
        ];

        let col = [
            -((right + left) / (right - left)),
            -((top + bottom) / (top - bottom)),
            -((z_far + z_near) / (z_far - z_near)),
            One::ONE,
        ];

        ortho.set_rightwards_diagonal(diag);
        ortho.set_col(3, col);

        ortho
    }
}

impl<T> Matrix4<T>
where
    T: Trig
        + One
        + Zero
        + Bounded
        + CheckedAddAssign
        + ClosedAdd
        + ClosedNeg
        + ClosedDiv
        + ClosedSub
        + ClosedMul,
{
    #[must_use]
    #[inline]
    pub fn perspective_3d(fov: Angle<T>, aspect: T, near: T, far: T) -> Self {
        // Taken from https://www.mauriciopoppe.com/notes/computer-graphics/viewing/projection-transform/
        let two = n::<T>(nz!(2));
        let fov = fov.in_radians();

        let top = near * Trig::tan(fov / two);
        let right = aspect * top;

        let z = T::ZERO;

        Matrix::new([
            [T::ONE / right, z, z, z],
            [z, T::ONE / top, z, z],
            [
                z,
                z,
                ((far + near) / (far - near)).neg(),
                (two.neg() * far * near) / (far - near),
            ],
            [z, z, T::ONE.neg(), z],
        ])
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where
    T: AddAssign
        + SubAssign
        + Zero
        + Copy
        + ClosedAdd
        + ClosedMul
        + ClosedSub
        + PartialEq
        + ClosedNeg
        + ClosedDiv
        + One,
{
    /// Calculates the inverse of the matrix.
    ///
    /// If you have a matrix `m1`, and `m2`, multiplying `m1` by `m2`, and then multiplying
    /// `m1` by `m2.inverse()` will return the original `m1`.
    ///
    /// # Panics
    ///
    /// This method will panic if the inverse cannot be computed. If the matrix is potentially
    /// not invertible, then use the [`inverse_checked()`] or [`has_inverse()`] methods to
    /// check for potential failures.
    ///
    /// [`inverse_checked()`]: ./struct.Matrix.html#method.inverse_checked
    /// [`has_inverse()`]: ./struct.Matrix.html#method.has_inverse
    #[track_caller]
    #[must_use]
    #[inline]
    pub fn inverse(self) -> Self {
        Matrix::inverse_checked(self)
            .expect("Could not calculate matrix inverse: determinant must not be '0'")
    }

    /// Calculates the inverse of the matrix.
    ///
    /// If you have a matrix `m1`, and `m2`, multiplying `m1` by `m2`, and then multiplying
    /// `m1` by `m2.inverse()` will return the original `m1`.
    ///
    /// If the matrix is not invertible, then this function will return `None`.
    #[must_use]
    #[inline]
    pub fn inverse_checked(self) -> Option<Self> {
        let det = self.determinant();
        if det == Zero::ZERO {
            return None;
        }

        Some(self.adjoint() * (T::ONE / det))
    }

    #[inline]
    pub fn invert(&mut self) {
        *self = self.inverse();
    }
}

impl<T> Matrix4<T>
where
    T: AddAssign
        + Abs
        + SubAssign
        + Zero
        + Copy
        + ClosedAdd
        + ClosedMul
        + ClosedSub
        + PartialEq
        + ClosedNeg
        + ClosedDiv
        + One
        + Sqrt
        + PartialOrd,
{
    /// Decompose a 3d homogeneous transform matrix into a tuple of `(translation, scale, rotation, w)`
    #[cfg(feature = "nightly")]
    #[doc(alias = "polar_decompose")]
    #[must_use]
    #[inline]
    pub fn decompose_homogeneous_transform_3d<R: Rotation<3, Scalar = T>>(
        mut self,
    ) -> (Vector3<T>, Vector3<T>, R, T) {
        let translation = Vector::new(self.col(3)).shrink_to();
        let w = self[3][3];

        // Remove the translation part from the matrix.
        self.set_col(3, Vector4::unit_n::<3>().to_array());

        // Polar decompose the matrix.
        let mut count = 0usize;
        let mut norm = T::ONE;
        let mut rot_mat = self;

        let half = T::ONE / (T::ONE + T::ONE);

        while count < 100 && norm > Zero::ZERO {
            let rot_trans = rot_mat.transpose();
            let rot_inv_trans = rot_trans.inverse();

            let rot_next = (rot_mat + rot_inv_trans) * half;

            norm = T::ZERO;
            for i in 0..3 {
                let lhs_row = shrink_to::<3, _, _>(rot_mat.row(i));
                let rhs_row = shrink_to::<3, _, _>(rot_next.row(i));

                let n = sum(zip_map(lhs_row, rhs_row, |lhs, rhs| (rhs - lhs).abs()));
                norm = max_by(norm, n, |x, y| x.partial_cmp(y).unwrap_or(Ordering::Less));
            }

            rot_mat = rot_next;
            count += 1;
        }

        let rotation = Rotation::from_homogeneous(rot_mat);
        let scale =
            Vector::new((self * rot_mat.transpose()).rightwards_diagonal()).shrink_to::<3>();

        (translation, scale, rotation, w)
    }
}

#[cfg(feature = "nightly")]
pub trait TransformHomogeneous<const DIM: usize> {
    type Scalar;

    #[must_use]
    fn transform_homogeneous(self, matrix: Matrix<Self::Scalar, { DIM + 1 }, { DIM + 1 }>) -> Self;
}

impl<T, const ROWS: usize, const COLS: usize> Mul<Vector<T, COLS>> for Matrix<T, ROWS, COLS>
where
    T: Zero + One + PartialEq + Copy + DivAssign + ClosedMul + ClosedAdd,
{
    type Output = Vector<T, ROWS>;
    #[inline]
    fn mul(self, rhs: Vector<T, COLS>) -> Self::Output {
        let result = self
            .data
            .map(|row| sum(zip_map(row, rhs.to_array(), Mul::mul)));
        Vector::from(result)
    }
}

impl<T, const ROWS: usize, const COLS: usize> Mul<Matrix<T, ROWS, COLS>> for Vector<T, COLS>
where
    T: Zero + One + PartialEq + Copy + DivAssign + ClosedMul + ClosedAdd,
{
    type Output = Vector<T, ROWS>;
    #[inline]
    fn mul(self, rhs: Matrix<T, ROWS, COLS>) -> Self::Output {
        rhs * self
    }
}

macro_rules! impl_matrix_conversion {
    ( row $matrix_name:ident => ($rows:literal, $cols:literal) [ $( $row_vecs:ident ),* $(,)? ] ) => {
        #[cfg(feature = "mint")]
        impl<T> From<mint:: $matrix_name <T>> for Matrix<T, $rows, $cols> {
            #[inline]
            fn from(value: mint:: $matrix_name<T>) -> Self {
                Matrix::new([
                    $( value. $row_vecs .into(), )*
                ])
            }
        }

        #[cfg(feature = "mint")]
        impl<T> From<Matrix<T, $rows, $cols>> for mint:: $matrix_name<T> {
            #[inline]
            fn from(value: Matrix<T, $rows, $cols>) -> Self {
                mint::$matrix_name::from(value.to_array())
            }
        }

        #[cfg(feature = "mint")]
        impl<T> mint::IntoMint for Matrix<T, $rows, $cols> {
            type MintType = mint:: $matrix_name<T>;
        }

        #[cfg(feature = "mint")]
        impl<T: PartialEq> PartialEq<mint::$matrix_name <T>> for Matrix<T, $rows, $cols> {
            #[inline]
            fn eq(&self, other: &mint::$matrix_name<T>) -> bool {
                let (lhs, rhs): (&[T], &[T; _]) = (self.as_ref(), other.as_ref());
                PartialEq::eq(lhs, rhs)
            }
        }

        #[cfg(feature = "mint")]
        impl<T: PartialEq> PartialEq<Matrix<T, $rows, $cols>> for mint::$matrix_name <T> {
            #[inline]
            fn eq(&self, other: &Matrix<T, $rows, $cols>) -> bool {
                let (lhs, rhs): (&[T; _], &[T]) = (self.as_ref(), other.as_ref());
                PartialEq::eq(lhs, rhs)
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::AbsDiffEq> approx::AbsDiffEq<mint::$matrix_name<T>>
            for Matrix<T, $rows, $cols>
        where
            T::Epsilon: Copy,
        {
            type Epsilon = T::Epsilon;

            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &mint::$matrix_name<T>, epsilon: Self::Epsilon) -> bool {
                self.as_slice()
                    .iter()
                    .zip(AsRef::<[T; $rows * $cols]>::as_ref(other))
                    .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::AbsDiffEq> approx::AbsDiffEq<Matrix<T, $rows, $cols>>
            for mint::$matrix_name<T>
        where
            T::Epsilon: Copy,
        {
            type Epsilon = T::Epsilon;

            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &Matrix<T, $rows, $cols>, epsilon: Self::Epsilon) -> bool {
                AsRef::<[T; $rows * $cols]>::as_ref(self)
                    .iter()
                    .zip(other.as_slice())
                    .all(|(x, y)| x.abs_diff_eq(y, epsilon))
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::RelativeEq> approx::RelativeEq<mint::$matrix_name<T>>
            for Matrix<T, $rows, $cols>
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
                other: &mint::$matrix_name<T>,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                self.as_slice()
                    .iter()
                    .zip(AsRef::<[T; $rows * $cols]>::as_ref(other))
                    .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::RelativeEq> approx::RelativeEq<Matrix<T, $rows, $cols>>
            for mint::$matrix_name<T>
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
                other: &Matrix<T, $rows, $cols>,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                AsRef::<[T; $rows * $cols]>::as_ref(self)
                    .iter()
                    .zip(other.as_slice())
                    .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::UlpsEq> approx::UlpsEq<mint::$matrix_name<T>>
            for Matrix<T, $rows, $cols>
        where
            T::Epsilon: Copy,
        {
            #[inline]
            fn default_max_ulps() -> u32 {
                T::default_max_ulps()
            }

            #[inline]
            fn ulps_eq(&self, other: &mint::$matrix_name<T>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                self.as_slice()
                    .iter()
                    .zip(AsRef::<[T; $rows * $cols]>::as_ref(other))
                    .all(|(x, y)| x.ulps_eq(y, epsilon, max_ulps))
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::UlpsEq> approx::UlpsEq<Matrix<T, $rows, $cols>>
            for mint::$matrix_name<T>
        where
            T::Epsilon: Copy,
        {
            #[inline]
            fn default_max_ulps() -> u32 {
                T::default_max_ulps()
            }

            #[inline]
            fn ulps_eq(&self, other: &Matrix<T, $rows, $cols>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                AsRef::<[T; $rows * $cols]>::as_ref(self)
                    .iter()
                    .zip(other.as_slice())
                    .all(|(x, y)| x.ulps_eq(y, epsilon, max_ulps))
            }
        }
    };

    ( col $matrix_name:ident => ($rows:literal, $cols:literal) [ $( $row_vecs:ident ),* $(,)? ] ) => {
        #[cfg(feature = "mint")]
        impl<T> From<mint:: $matrix_name <T>> for Matrix<T, $rows, $cols> {
            #[inline]
            fn from(value: mint:: $matrix_name<T>) -> Self {
                Matrix::new([$( value. $row_vecs .into(), )*]).transpose()
            }
        }

        #[cfg(feature = "mint")]
        impl<T> From<Matrix<T, $rows, $cols>> for mint:: $matrix_name<T> {
            #[inline]
            fn from(value: Matrix<T, $rows, $cols>) -> Self {
                mint::$matrix_name::from(value.transpose().to_array())
            }
        }

        #[cfg(feature = "mint")]
        impl<T: PartialEq> PartialEq<mint::$matrix_name <T>> for Matrix<T, $rows, $cols> {
            #[inline]
            fn eq(&self, other: &mint::$matrix_name<T>) -> bool {
                let cols: &[[T; $rows]; $cols] = other.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(self.col_ref(i)) {
                        if !PartialEq::eq(e1, e2) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        #[cfg(feature = "mint")]
        impl<T: PartialEq> PartialEq<Matrix<T, $rows, $cols>> for mint::$matrix_name <T> {
            #[inline]
            fn eq(&self, other: &Matrix<T, $rows, $cols>) -> bool {
                let cols: &[[T; $rows]; $cols] = self.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(other.col_ref(i)) {
                        if !PartialEq::eq(e1, e2) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::AbsDiffEq> approx::AbsDiffEq<mint::$matrix_name<T>>
            for Matrix<T, $rows, $cols>
        where
            T::Epsilon: Copy,
        {
            type Epsilon = T::Epsilon;

            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &mint::$matrix_name<T>, epsilon: Self::Epsilon) -> bool {
                let cols: &[[T; $rows]; $cols] = other.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(self.col_ref(i)) {
                        if !e1.abs_diff_eq(e2, epsilon) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::AbsDiffEq> approx::AbsDiffEq<Matrix<T, $rows, $cols>>
            for mint::$matrix_name<T>
        where
            T::Epsilon: Copy,
        {
            type Epsilon = T::Epsilon;

            #[inline]
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }

            #[inline]
            fn abs_diff_eq(&self, other: &Matrix<T, $rows, $cols>, epsilon: Self::Epsilon) -> bool {
                let cols: &[[T; $rows]; $cols] = self.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(other.col_ref(i)) {
                        if !e1.abs_diff_eq(e2, epsilon) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::RelativeEq> approx::RelativeEq<mint::$matrix_name<T>>
            for Matrix<T, $rows, $cols>
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
                other: &mint::$matrix_name<T>,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                let cols: &[[T; $rows]; $cols] = other.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(self.col_ref(i)) {
                        if !e1.relative_eq(e2, epsilon, max_relative) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::RelativeEq> approx::RelativeEq<Matrix<T, $rows, $cols>>
            for mint::$matrix_name<T>
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
                other: &Matrix<T, $rows, $cols>,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                let cols: &[[T; $rows]; $cols] = self.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(other.col_ref(i)) {
                        if !e1.relative_eq(e2, epsilon, max_relative) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::UlpsEq> approx::UlpsEq<mint::$matrix_name<T>>
            for Matrix<T, $rows, $cols>
        where
            T::Epsilon: Copy,
        {
            #[inline]
            fn default_max_ulps() -> u32 {
                T::default_max_ulps()
            }

            #[inline]
            fn ulps_eq(&self, other: &mint::$matrix_name<T>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                let cols: &[[T; $rows]; $cols] = other.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(self.col_ref(i)) {
                        if !e1.ulps_eq(e2, epsilon, max_ulps) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        #[cfg(all(feature = "approx", feature = "mint"))]
        impl<T: approx::UlpsEq> approx::UlpsEq<Matrix<T, $rows, $cols>>
            for mint::$matrix_name<T>
        where
            T::Epsilon: Copy,
        {
            #[inline]
            fn default_max_ulps() -> u32 {
                T::default_max_ulps()
            }

            #[inline]
            fn ulps_eq(&self, other: &Matrix<T, $rows, $cols>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                let cols: &[[T; $rows]; $cols] = self.as_ref();
                for i in 0..$cols {
                    for (e1, e2) in cols[i].as_ref().into_iter().zip(other.col_ref(i)) {
                        if !e1.ulps_eq(e2, epsilon, max_ulps) {
                            return false;
                        }
                    }
                }
                true
            }
        }
    };
}

macro_rules! impl_matrix_conversions {
    (
        $(
            $major_ty:ident $matrix_name:ident => ($rows:literal, $cols:literal) [ $( $row_vecs:ident ),* $(,)? ]
        )*
    ) => {
        $(
            impl_matrix_conversion!( $major_ty $matrix_name => ($rows, $cols) [ $( $row_vecs ),* ]);
        )*
    };
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Zeroable, const ROWS: usize, const COLS: usize> bytemuck::Zeroable
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn zeroed() -> Self {
        Matrix::from_fn(|_, _| bytemuck::Zeroable::zeroed())
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Pod, const ROWS: usize, const COLS: usize> bytemuck::Pod
    for Matrix<T, ROWS, COLS>
{
}

#[cfg(feature = "matrixcompare")]
impl<T: Copy, const ROWS: usize, const COLS: usize> matrixcompare_core::Matrix<T>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn rows(&self) -> usize {
        ROWS
    }

    #[inline]
    fn cols(&self) -> usize {
        COLS
    }

    #[inline]
    fn access(&'_ self) -> matrixcompare_core::Access<'_, T> {
        matrixcompare_core::Access::Dense(self)
    }
}

#[cfg(feature = "matrixcompare")]
impl<T: Copy, const ROWS: usize, const COLS: usize> matrixcompare_core::DenseAccess<T>
    for Matrix<T, ROWS, COLS>
{
    #[inline]
    fn fetch_single(&self, row: usize, col: usize) -> T {
        self[row][col]
    }
}

#[cfg(feature = "approx")]
impl<T: approx::AbsDiffEq, const ROWS: usize, const COLS: usize> approx::AbsDiffEq
    for Matrix<T, ROWS, COLS>
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
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .all(|(x, y)| x.abs_diff_eq(y, epsilon))
    }
}

#[cfg(feature = "approx")]
impl<T: approx::RelativeEq, const ROWS: usize, const COLS: usize> approx::RelativeEq
    for Matrix<T, ROWS, COLS>
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
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .all(|(x, y)| x.relative_eq(y, epsilon, max_relative))
    }
}

#[cfg(feature = "approx")]
impl<T: approx::UlpsEq, const ROWS: usize, const COLS: usize> approx::UlpsEq
    for Matrix<T, ROWS, COLS>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.as_slice()
            .iter()
            .zip(other.as_slice())
            .all(|(x, y)| x.ulps_eq(y, epsilon, max_ulps))
    }
}

impl_matrix_conversions! {
    row RowMatrix2 => (2, 2) [x, y]
    row RowMatrix3 => (3, 3) [x, y, z]
    row RowMatrix4 => (4, 4) [x, y, z, w]

    row RowMatrix2x3 => (2, 3) [x, y]
    row RowMatrix2x4 => (2, 4) [x, y]
    row RowMatrix3x2 => (3, 2) [x, y, z]
    row RowMatrix3x4 => (3, 4) [x, y, z]
    row RowMatrix4x2 => (4, 2) [x, y, z, w]
    row RowMatrix4x3 => (4, 3) [x, y, z, w]

    col ColumnMatrix2 => (2, 2) [x, y]
    col ColumnMatrix3 => (3, 3) [x, y, z]
    col ColumnMatrix4 => (4, 4) [x, y, z, w]

    col ColumnMatrix2x3 => (2, 3) [x, y, z]
    col ColumnMatrix2x4 => (2, 4) [x, y, z, w]
    col ColumnMatrix3x2 => (3, 2) [x, y]
    col ColumnMatrix3x4 => (3, 4) [x, y, z, w]
    col ColumnMatrix4x2 => (4, 2) [x, y]
    col ColumnMatrix4x3 => (4, 3) [x, y, z]
}
