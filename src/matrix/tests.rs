// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(any(feature = "std", feature = "libm"))]
use crate::rotation::angle::Angle;
#[cfg(feature = "simd")]
use crate::simd::{SimdMul, SimdValue};
#[cfg(feature = "nightly")]
use crate::{matrix::TransformHomogeneous, point::Point3, utils::num::Zero};
use crate::{
    matrix::{Matrix, Matrix4},
    rotation::quaternion::Quaternion,
    vector::{Vector, Vector3},
};
#[cfg(all(any(feature = "std", feature = "libm"), feature = "nightly"))]
use core::ops::Neg;

#[test]
fn test_matrix_access() {
    #[rustfmt::skip]
    let mut matrix = Matrix::new([
        [01, 02, 03, 04, 05],
        [06, 07, 08, 09, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [12, 22, 23, 24, 25],
    ]);

    assert_eq!(matrix.row(2), [11, 12, 13, 14, 15]);
    assert_eq!(matrix.col(4), [05, 10, 15, 20, 25]);
    assert_eq!(matrix.rightwards_diagonal(), [01, 07, 13, 19, 25]);
    assert_eq!(matrix.leftwards_diagonal(), [05, 09, 13, 17, 12]);

    for i in 0..5 {
        assert_eq!(matrix.row_ref(i).map(|elem| *elem), matrix.row(i));
        assert_eq!(matrix.row_mut(i).map(|elem| *elem), matrix.row(i));

        assert_eq!(matrix.col_ref(i).map(|elem| *elem), matrix.col(i));
        assert_eq!(matrix.col_mut(i).map(|elem| *elem), matrix.col(i));
    }

    assert_eq!(
        matrix.leftwards_diagonal_ref().map(|elem| *elem),
        matrix.leftwards_diagonal(),
    );
    assert_eq!(
        matrix.leftwards_diagonal_mut().map(|elem| *elem),
        matrix.leftwards_diagonal(),
    );

    assert_eq!(
        matrix.rightwards_diagonal_ref().map(|elem| *elem),
        matrix.rightwards_diagonal(),
    );
    assert_eq!(
        matrix.rightwards_diagonal_mut().map(|elem| *elem),
        matrix.rightwards_diagonal(),
    );

    {
        let matrix_ref = matrix.each_ref();
        assert_eq!(*matrix_ref[0][0], 1);

        let mut matrix_mut = matrix.each_mut();
        *matrix_mut[0][0] = 99;
    }

    assert_eq!(matrix[0][0], 99);
}

#[cfg(feature = "nightly")]
#[test]
fn test_concat() {
    let v1 = Vector::new([1, 2, 3, 4]);
    let v2 = Vector::new([5, 6, 7]);

    assert_eq!(v1.concat(v2).to_array(), [1, 2, 3, 4, 5, 6, 7]);

    #[rustfmt::skip]
    let m1 = Matrix::new([
        [01, 02, 03],
        [04, 05, 06],
        [07, 08, 09],
    ]);

    #[rustfmt::skip]
    let m2 = Matrix::new([
        [11, 12, 13],
        [14, 15, 16],
        [17, 18, 19],
    ]);

    #[rustfmt::skip]
    assert_eq!(m1.concat_horizontal(m2), Matrix::new([
        [01, 02, 03, 11, 12, 13],
        [04, 05, 06, 14, 15, 16],
        [07, 08, 09, 17, 18, 19],
    ]));

    #[rustfmt::skip]
    assert_eq!(m1.concat_vertical(m2), Matrix::new([
        [01, 02, 03],
        [04, 05, 06],
        [07, 08, 09],
        [11, 12, 13],
        [14, 15, 16],
        [17, 18, 19],
    ]));
}

#[test]
fn test_matrix_multiply() {
    #[rustfmt::skip]
    let m1 = Matrix4::new([
        [15, 07, 09, 10],
        [02, 03, 03, 08],
        [08, 10, 02, 03],
        [03, 03, 04, 08],
    ]);

    #[rustfmt::skip]
    let m2 = Matrix4::new([
        [03, 10, 12, 18],
        [12, 01, 04, 09],
        [09, 10, 12, 02],
        [03, 12, 04, 10],
    ]);

    #[rustfmt::skip]
    let result = Matrix4::new([
        [240, 367, 356, 451],
        [093, 149, 104, 149],
        [171, 146, 172, 268],
        [105, 169, 128, 169],
    ]);

    assert_eq!(m1 * m2, result);
    #[cfg(feature = "simd")]
    {
        assert_eq!(SimdValue(m1) * SimdValue(m2), SimdValue(result));
    }

    assert_eq!(
        Matrix4::<f32>::identity() * Matrix4::identity(),
        Matrix4::identity()
    );

    let m0 = Matrix::from_row_vector(From::from([1, 2, 3]));
    let m1 = Matrix::from_column_vector(From::from([4, 5, 6]));

    assert_eq!(m0 * m1, Matrix::new([[32]]));

    #[rustfmt::skip]
    let m1_by_m0_result = Matrix::new([
        [04, 08, 12],
        [05, 10, 15],
        [06, 12, 18],
    ]);

    assert_eq!(m1 * m0, m1_by_m0_result);

    #[cfg(feature = "simd")]
    {
        assert_eq!(m1.simd_mul(m0), m1_by_m0_result);
    }
}

#[test]
fn test_matrix_add() {
    // Taken from https://en.wikipedia.org/wiki/Matrix_addition
    #[rustfmt::skip]
    let m1 = Matrix::new([
        [1, 3],
        [1, 0],
        [1, 2],
    ]);

    #[rustfmt::skip]
    let m2 = Matrix::new([
        [0, 0],
        [7, 5],
        [2, 1],
    ]);

    #[rustfmt::skip]
    let expected_add_result = Matrix::new([
        [1, 3],
        [8, 5],
        [3, 3],
    ]);

    #[rustfmt::skip]
    let expected_sub_result = Matrix::new([
        [01, 03],
        [-6, -5],
        [-1, 01],
    ]);

    assert_eq!(m1 + m2, expected_add_result);
    assert_eq!(m1 - m2, expected_sub_result);
}

#[cfg(feature = "nightly")]
#[test]
fn test_matrix_transform() {
    let origin = Point3::new([1.0, 2.0, 3.0]);
    let offset = Vector3::new([1.0, 1.0, 1.0]);
    let origin_homogeneous = origin.expand(1.0).into_vector();

    let translation = Matrix::translation(offset);

    assert_eq!(
        origin + offset,
        (translation * origin_homogeneous).shrink().into_point()
    );
    assert_eq!(origin + offset, Point3::new([2.0, 3.0, 4.0]));

    assert_eq!(origin + offset, origin.transform_homogeneous(translation));

    let rotation = Matrix::<f64, _, _>::rotation_3d(Quaternion::identity());
    assert_eq!(rotation, Matrix::identity());

    approx::assert_relative_eq!(
        Matrix4::<f64>::identity(),
        Matrix4::<f64>::identity() * Matrix4::<f64>::identity() * Matrix4::<f64>::identity()
    );
}

#[cfg(any(feature = "std", feature = "libm"))]
#[cfg_attr(miri, ignore = "🐌 takes too long by default on miri")]
#[test]
fn test_rotation_matrices() {
    use approx::assert_relative_eq;

    let mut angle = Angle::<f64>::zero();
    while angle < Angle::full() {
        let matrix = Matrix::x_axis_rotation(angle);
        let matrix_2 = Matrix::axis_rotation_3d(angle, Vector3::X);

        approx::assert_relative_eq!(matrix, matrix_2, epsilon = 1e-15);
        angle += Angle::Degrees(1.0);
    }

    let mut angle = Angle::<f64>::zero();
    while angle < Angle::full() {
        let matrix = Matrix::y_axis_rotation(angle);
        let matrix_2 = Matrix::axis_rotation_3d(angle, Vector3::Y);

        approx::assert_relative_eq!(matrix, matrix_2, epsilon = 1e-15);
        angle += Angle::Degrees(1.0);
    }

    let mut angle = Angle::<f64>::zero();
    while angle < Angle::full() {
        let matrix = Matrix::z_axis_rotation(angle);
        let matrix_2 = Matrix::axis_rotation_3d(angle, Vector3::Z);

        approx::assert_relative_eq!(matrix, matrix_2, epsilon = 1e-15);
        angle += Angle::Degrees(1.0);
    }

    let test_matrix_quaternion_conversion = |axis: Vector3<f64>| {
        let mut angle = Angle::<f64>::zero();
        while angle < Angle::full() {
            let quat = Quaternion::from_angle_axis(angle, axis);
            let matrix = Matrix::axis_rotation_3d(angle, axis);
            let matrix_2 = Matrix::rotation_3d(quat);

            approx::assert_relative_eq!(matrix, matrix_2, epsilon = 1e-15);

            angle += Angle::Degrees(1.0);
        }
    };

    test_matrix_quaternion_conversion(Vector::X);
    test_matrix_quaternion_conversion(Vector::Y);
    test_matrix_quaternion_conversion(Vector::Z);

    let quat = Quaternion::from_angle_axis(
        Angle::Degrees(21.0),
        Vector3::new([0.7, 0.3, 0.4]).normalized(),
    );
    let mat = Matrix::rotation_3d(quat);
    let back_to_quat = Quaternion::from_matrix(mat);

    assert_relative_eq!(quat, back_to_quat, epsilon = 1e-15);
}

#[cfg(feature = "nightly")]
#[test]
fn test_uniform_scaling() {
    let point = Point3::splat(1.0);
    let scale_factor = 3.0;

    let scaling = Matrix::scaling(Vector3::splat(scale_factor));
    assert_eq!(
        Point3::splat(scale_factor),
        (scaling * point.expand(1.0).into_vector())
            .shrink()
            .into_point()
    );
}

#[test]
fn test_matrix_resize() {
    #[rustfmt::skip]
    let m1 = Matrix::new([
        [1],
        [2],
        [3],
        [4],
    ]);

    #[rustfmt::skip]
    let m2 = Matrix::new([
        [1, 2],
        [3, 4],
    ]);

    assert_eq!(m1.resize::<2, 2>(), m2);
}

#[test]
fn test_transpose() {
    #[rustfmt::skip]
    let mut mat = Matrix::new([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]);

    #[rustfmt::skip]
    let transposed = Matrix::new([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    ]);

    assert_eq!(mat.transpose(), transposed);
    mat.transpose_in_place();
    assert_eq!(mat, transposed);

    #[rustfmt::skip]
    let mat = Matrix::new([
        [1, 2],
        [3, 4],
        [5, 6],
    ]);
    #[rustfmt::skip]
    let transposed = Matrix::new([
        [1, 3, 5],
        [2, 4, 6],
    ]);

    assert_eq!(mat.transpose(), transposed);
    assert_eq!(mat.transpose().transpose(), mat);

    let mut mat = Matrix::new([[1]]);
    mat.transpose_in_place();
    assert_eq!(mat[0][0], 1);

    let mat = Matrix::new([[1, 2, 3, 4]]);
    assert_eq!(mat.transpose(), Matrix::new([[1], [2], [3], [4]]));
}

#[test]
fn test_determinant() {
    let mat = Matrix::new([[25]]);
    assert_eq!(mat.determinant(), 25);

    // 2x2 example taken from https://www.mathsisfun.com/algebra/matrix-determinant.html
    #[rustfmt::skip]
    let mat = Matrix::new([
        [3, 8],
        [4, 6],
    ]);

    assert_eq!(mat.determinant(), -14);

    // 3x3 examples taken from https://www.geeksforgeeks.org/maths/determinant-of-3x3-matrix/
    #[rustfmt::skip]
    let mat = Matrix::new([
        [1, 2, 1],
        [0, 3, 0],
        [4, 1, 2],
    ]);

    assert_eq!(mat.determinant(), -6);

    #[rustfmt::skip]
    let mat = Matrix::new([
        [3, 1, 2],
        [0, 2, 5],
        [2, 0, 4],
    ]);

    assert_eq!(mat.determinant(), 26);

    #[rustfmt::skip]
    let mat = Matrix::new([
        [00, -1, 2],
        [03, 02, 0],
        [-1, 03, 2],
    ]);

    assert_eq!(mat.determinant(), 28);

    #[rustfmt::skip]
    let mat = Matrix::new([
        [02, 01, 3, 4],
        [00, -1, 2, 1],
        [03, 02, 0, 5],
        [-1, 03, 2, 1],
    ]);

    assert_eq!(mat.determinant(), 35);

    #[rustfmt::skip]
    let mat = Matrix::new([
        [02, 01, 00, 3],
        [04, -1, 02, 0],
        [-3, 02, 01, 5],
        [01, 00, -2, 3],
    ]);

    assert_eq!(mat.determinant(), -85);

    assert_eq!(Matrix::<i32, 4, 4>::identity().determinant(), 1);
}

#[cfg(feature = "nightly")]
#[test]
fn test_cofactor() {
    #[rustfmt::skip]
    let mat = Matrix::new([
        [1, 2],
        [3, 4],
    ]);

    assert_eq!(mat.cofactor(0, 0), Matrix::new([[4]]));
    assert_eq!(mat.cofactor(0, 1), Matrix::new([[3]]));
    assert_eq!(mat.cofactor(1, 0), Matrix::new([[2]]));
    assert_eq!(mat.cofactor(1, 1), Matrix::new([[1]]));
}

#[test]
fn test_adjoint() {
    #[rustfmt::skip]
    let mat = Matrix::new([
        [03, 6],
        [-4, 8],
    ]);

    #[rustfmt::skip]
    let expected_adjoint = Matrix::new([
        [8, -6],
        [4, 03],
    ]);

    assert_eq!(mat.adjoint(), expected_adjoint);

    #[rustfmt::skip]
    let mat = Matrix::new([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]);

    #[rustfmt::skip]
    let expected_adjoint = Matrix::new([
        [-3, 006, -3],
        [06, -12, 06],
        [-3, 006, -3],
    ]);

    assert_eq!(mat.adjoint(), expected_adjoint);

    #[rustfmt::skip]
    let mat = Matrix::new([
        [05, -2, 02, 7],
        [01, 00, 00, 3],
        [-3, 01, 05, 0],
        [03, -1, -9, 4],
    ]);

    #[rustfmt::skip]
    let expected_adjoint = Matrix::new([
        [-12, 076, -60, -36],
        [-56, 208, -82, -58],
        [004, 004, -02, -10],
        [004, 004, 020, 012],
    ]);

    assert_eq!(mat.adjoint(), expected_adjoint);
}

#[cfg(feature = "nightly")]
#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn test_inverse() {
    let translation_v = Vector3::new([1.0, 2.0, 3.0]);
    let translation_m = Matrix4::translation(translation_v);

    let inverse_translation_v = translation_v.neg();
    let inverse_translation_m = Matrix4::translation(inverse_translation_v);

    assert_eq!(translation_m.inverse(), inverse_translation_m);
    assert_eq!(translation_m, inverse_translation_m.inverse());

    let scale_v = Vector3::new([4.0, 6.0, 8.0]);
    let scale_m = Matrix4::scaling(scale_v);

    let inverse_scale_v = scale_v.map(|elem| 1.0 / elem);
    let inverse_scale_m = Matrix4::scaling(inverse_scale_v);

    assert_eq!(scale_m.inverse(), inverse_scale_m);
    assert_eq!(scale_m, inverse_scale_m.inverse());

    assert_eq!(Matrix4::<f64>::ZERO.inverse_checked(), None);

    let rotation = Matrix::x_axis_rotation(Angle::<f64>::half());
    let inverse = rotation.inverse();

    approx::assert_relative_eq!(rotation, inverse, epsilon = 1e-15);
}

#[cfg(feature = "nightly")]
#[cfg(any(feature = "std", feature = "libm"))]
#[cfg_attr(miri, ignore = "🐌 takes too long by default on miri")]
#[test]
fn test_decompose() {
    let rot = Quaternion::from_angle_axis(
        Angle::Degrees(21.0),
        Vector3::new([0.7, 0.3, 0.4]).normalized(),
    );
    let scale = Vector::new([2.0, 3.0, 4.0]);
    let translation = Vector::new([14.0, 23.0, 196.0]);

    let trans_mat = Matrix::translation(translation);
    let scale_mat = Matrix::scaling(scale);
    let rot_mat = Matrix::rotation_3d(rot);

    let mat = trans_mat * scale_mat * rot_mat;

    let (decomp_translation, decomp_scale, decomp_rot, w) =
        mat.decompose_homogeneous_transform_3d::<Quaternion<_>>();
    assert_eq!(translation, decomp_translation);
    approx::assert_abs_diff_eq!(rot.into_vector(), decomp_rot.into_vector(), epsilon = 1e-15);
    approx::assert_abs_diff_eq!(scale, decomp_scale, epsilon = 1e-15);
    assert_eq!(w, 1.0);
}

#[test]
fn test_mint_conversions() {
    use mint::ColumnMatrix2x3;

    let mint_matrix = ColumnMatrix2x3 {
        x: [1.0, 2.0].into(),
        y: [3.0, 4.0].into(),
        z: [5.0, 6.0].into(),
    };

    let matrix: Matrix<f64, 2, 3> = mint_matrix.into();

    assert_eq!(matrix, Matrix::new([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0],]));

    assert_eq!(matrix, mint_matrix);
}

#[cfg(feature = "simd")]
#[test]
fn test_simd() {
    let matrix = Matrix::new([[1.0f32, 2.0, 3.0, 4.0], [9.0, 8.0, 7.0, 6.0]]);

    let matrix = SimdValue(matrix);
    let result = matrix * SimdValue(2.0);

    assert_eq!(
        *result,
        Matrix::new([[2.0f32, 4.0, 6.0, 8.0], [18.0, 16.0, 14.0, 12.0],])
    )
}
