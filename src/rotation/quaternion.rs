use crate::utils::{
    num::{
        Bounded, ClosedAdd, ClosedDiv, ClosedMul, ClosedNeg, ClosedSub, One, Sqrt, Trig, Zero,
        checked::{CheckedAddAssign, CheckedDiv, CheckedMul},
        lerp, rat,
    },
    shrink_to, sum, zip,
};
use crate::{
    matrix::{Matrix4, TransformHomogeneous},
    point::Point,
    rotation::Rotation,
    rotation::angle::Angle,
    vector::{Vector, Vector3, Vector4},
};
use core::{
    array,
    cmp::{self, Ordering},
    convert::identity,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    ptr,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Quaternion<T> {
    pub v: Vector3<T>,
    pub w: T,
}

impl<T> Quaternion<T> {
    #[must_use]
    #[inline]
    pub fn from_components<V: Into<Vector3<T>>>(v: V, w: T) -> Self {
        Self { v: v.into(), w }
    }

    #[must_use]
    #[inline]
    pub const fn new(x: T, y: T, z: T, w: T) -> Self {
        Self {
            v: Vector3::new([x, y, z]),
            w,
        }
    }

    #[must_use]
    #[inline]
    pub const fn into_vector(self) -> Vector4<T> {
        let v = unsafe { ptr::read(&self.v) };
        let w = unsafe { ptr::read(&self.w) };

        let _self = ManuallyDrop::new(self);
        v.expand(w)
    }

    #[must_use]
    #[inline]
    pub const fn from_vector(vector: Vector4<T>) -> Self {
        let mut vector_part = [const { MaybeUninit::uninit() }; 3];
        unsafe {
            ptr::copy(vector.as_ptr(), vector_part.as_mut_ptr().cast(), 3);
        }

        let scalar = unsafe { ptr::read(vector.get_unchecked(3)) };

        let _vector = ManuallyDrop::new(vector);

        let vector = unsafe { Vector3::new(MaybeUninit::array_assume_init(vector_part)) };

        Self {
            v: vector,
            w: scalar,
        }
    }
}

impl<T> Quaternion<T> {
    #[must_use]
    #[inline]
    pub fn dot<U>(self, rhs: Quaternion<U>) -> T::Output
    where
        T: Mul<U>,
        T::Output: Zero + ClosedAdd,
    {
        Vector::dot(self.v, rhs.v) + (self.w * rhs.w)
    }
}

impl<T: Trig + One + ClosedAdd + ClosedDiv + ClosedMul + ClosedSub> Quaternion<T> {
    #[must_use]
    #[inline]
    pub fn from_euler_angles(yaw: Angle<T>, pitch: Angle<T>, roll: Angle<T>) -> Self {
        // Taken from:
        // https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
        let two = T::ONE + T::ONE;
        let (ys, yc) = (yaw.in_radians() / two).sin_cos();
        let (ps, pc) = (pitch.in_radians() / two).sin_cos();
        let (rs, rc) = (roll.in_radians() / two).sin_cos();

        let w = (yc * pc * rc) - (ys * ps * rs);
        let v = [
            (yc * pc * rs) + (ys * ps * rc),
            (ys * pc * rc) + (yc * ps * rs),
            (yc * ps * rc) + (ys * pc * rs),
        ];

        Self::from_components(v, w)
    }
}

impl<T: Trig + One + ClosedDiv + ClosedAdd + ClosedMul> Quaternion<T> {
    #[must_use]
    #[inline]
    pub fn from_angle_axis(angle: Angle<T>, axis: Vector3<T>) -> Self {
        // Taken from:
        // https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        let half_angle = angle.in_radians() / (T::ONE + T::ONE);
        let (sin_a, cos_a) = Trig::sin_cos(half_angle);

        Quaternion::from_components(axis * sin_a, cos_a)
    }
}

impl<T: Trig + One + ClosedDiv + ClosedSub + ClosedAdd + ClosedMul + Sqrt> Quaternion<T> {
    #[must_use]
    #[inline]
    pub fn into_angle_axis(self) -> (Angle<T>, Vector3<T>) {
        // Taken from:
        // https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm
        let two = T::ONE + T::ONE;
        let qw = self.w;
        let qw_sq = qw * qw;

        let angle = Angle::Radians(two * qw.acos());
        let axis = self.v.map(|elem| elem / (T::ONE - qw_sq));

        (angle, axis)
    }
}

impl<T: ClosedNeg> Quaternion<T> {
    #[must_use]
    #[inline]
    pub fn conjugated(self) -> Self {
        Quaternion::from_components(Neg::neg(self.v), self.w)
    }

    #[inline]
    pub fn conjugate(&mut self) {
        unsafe {
            let vec = Neg::neg(ptr::read(&self.v));
            ptr::write(&mut self.v, vec);
        }
    }
}

impl<T> Quaternion<T>
where
    T: ClosedMul + Copy + ClosedAdd + Div<<T as Add>::Output> + Zero + Sqrt,
{
    #[must_use]
    #[inline]
    pub fn normalized_unchecked(self) -> Quaternion<<T as Div>::Output> {
        self / Self::dot(self, self).sqrt()
    }
}

impl<T> Quaternion<T>
where
    T: Trig
        + Bounded
        + One
        + ClosedAdd
        + CheckedDiv<Output = T>
        + ClosedMul
        + ClosedSub
        + Neg<Output = T>
        + PartialOrd
        + CheckedAddAssign
        + Sqrt
        + Zero,
{
    #[must_use]
    #[inline]
    pub fn slerp(self, target: Quaternion<T>, time: T) -> Self {
        let cos_theta = Quaternion::dot(self, target);

        if cos_theta > rat(nz!(9995), nz!(10000)) {
            lerp(self, target, time).normalized()
        } else {
            fn clamp<T: PartialOrd>(val: T, min: T, max: T) -> T {
                let f = |x: &T, y: &T| x.partial_cmp(y).unwrap_or(Ordering::Less);
                cmp::min_by(cmp::max_by(val, min, f), max, f)
            }

            let theta = clamp(cos_theta, T::ONE.neg(), T::ONE).acos();
            let theta_p = theta * time;
            let perpendicular = (target - self) * cos_theta;
            let (p_cos, p_sin) = theta_p.sin_cos();

            (self * p_cos) + (perpendicular * p_sin)
        }
    }
}

impl<T> Quaternion<T>
where
    T: ClosedMul + Copy + ClosedAdd + CheckedDiv<Output = T> + Zero + Sqrt,
{
    #[must_use]
    #[inline]
    pub fn normalized(self) -> Quaternion<T> {
        self.normalized_checked().unwrap_or(self)
    }

    #[must_use]
    #[inline]
    pub fn normalized_checked(self) -> Option<Quaternion<T>> {
        self.checked_div(Self::dot(self, self).sqrt())
    }
}

impl<T: ClosedMul + Copy + ClosedAdd + DivAssign<<T as Add>::Output> + Zero + Sqrt> Quaternion<T> {
    #[inline]
    pub fn normalize(&mut self) {
        let rhs = Self::dot(*self, *self).sqrt();
        self.div_assign(rhs);
    }
}

impl<T> Quaternion<T>
where
    T: ClosedAdd + Copy + ClosedDiv + ClosedMul + ClosedSub + One + PartialOrd + Zero + Sqrt,
{
    #[must_use]
    #[inline]
    pub fn from_matrix(homogeneous_transform: Matrix4<T>) -> Self {
        let trace = sum(homogeneous_transform
            .rightwards_diagonal()
            .into_iter()
            .take(3));

        let m = &homogeneous_transform;

        if trace > T::ZERO {
            let two = T::ONE + T::ONE;
            let half = T::ONE / two;
            let mut s = (trace + T::ONE).sqrt();
            let w = s / two;
            s = half / s;

            let v = [
                (m[1][2] - m[2][1]) * s,
                (m[2][0] - m[0][2]) * s,
                (m[0][1] - m[1][0]) * s,
            ];

            Quaternion::from_components(v, w)
        } else {
            let mut diag =
                shrink_to::<3, _, _>(zip(m.rightwards_diagonal(), array::from_fn(identity)));

            // Sort by value
            {
                #[inline(always)]
                fn compare<T: PartialOrd>(lhs: &T, rhs: &T) -> Ordering {
                    lhs.partial_cmp(rhs).unwrap_or(Ordering::Less)
                }

                if compare(&diag[0].0, &diag[1].0) == Ordering::Less {
                    diag.swap(0, 1);
                }

                if compare(&diag[1].0, &diag[2].0) == Ordering::Less {
                    diag.swap(1, 2);
                }

                if compare(&diag[0].0, &diag[2].0) == Ordering::Less {
                    diag.swap(0, 2);
                }
            }

            let [(mut qi, i), (mut qj, j), (mut qk, k)] = diag;

            let half = T::ONE / (T::ONE + T::ONE);

            let mut s = (qi - (qj + qk) + T::ONE).sqrt();
            qi = s * half;

            if s != T::ZERO {
                s = half / s;
            }

            let w = (m[k][j] - m[j][k]) * s;

            qj = (m[j][i] + m[i][j]) * s;
            qk = (m[k][i] + m[i][k]) * s;

            Quaternion::from_components([qi, qj, qk], w)
        }
    }
}

impl<T> Rotation<3> for Quaternion<T>
where
    T: Bounded
        + CheckedDiv
        + CheckedAddAssign
        + Trig
        + ClosedNeg
        + ClosedAdd
        + ClosedDiv
        + ClosedMul
        + ClosedSub
        + Copy
        + DivAssign
        + One
        + PartialOrd
        + Sqrt
        + Zero,
{
    type Scalar = T;

    #[inline]
    fn identity() -> Self {
        <Quaternion<_>>::identity()
    }

    #[inline]
    fn inverse(self) -> Self {
        Quaternion::conjugated(self)
    }

    fn slerp(self, target: Self, time: Self::Scalar) -> Self {
        <Quaternion<_>>::slerp(self, target, time)
    }

    #[inline]
    fn transform_point(&self, point: Point<Self::Scalar, 3>) -> Point<Self::Scalar, 3> {
        point.transform_homogeneous(Matrix4::rotation_3d(*self))
    }

    #[inline]
    fn transform_vector(&self, vector: Vector<Self::Scalar, 3>) -> Vector<Self::Scalar, 3> {
        vector.transform_homogeneous(Matrix4::rotation_3d(*self))
    }

    #[inline]
    fn from_homogeneous(matrix: Matrix4<Self::Scalar>) -> Self {
        Self::from_matrix(matrix)
    }

    #[inline]
    fn into_homogeneous(self) -> Matrix4<Self::Scalar> {
        Matrix4::rotation_3d(self)
    }

    #[inline]
    fn get_homogeneous(&self) -> Matrix4<Self::Scalar> {
        Matrix4::rotation_3d(*self)
    }
}

impl<T: PartialOrd + ClosedSub> Quaternion<T> {
    #[must_use]
    #[inline]
    pub fn is_nearly_equal(self, to: Quaternion<T>, epsilon: T) -> bool {
        (self.into_vector() - to.into_vector())
            .into_iter()
            .all(|elem| elem < epsilon)
    }
}

impl<T: ClosedMul + Zero + ClosedAdd + ClosedDiv + ClosedSub + Sqrt + Copy> Mul<Quaternion<T>>
    for Quaternion<T>
{
    type Output = Quaternion<T>;
    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        Quaternion {
            v: Vector::cross(self.v, rhs.v) + Mul::mul(rhs.v, self.w) + Mul::mul(self.v, rhs.w),
            w: self.w * rhs.w - Vector::dot(self.v, rhs.v),
        }
    }
}

impl<T: Add> Add<Quaternion<T>> for Quaternion<T> {
    type Output = Quaternion<T::Output>;
    #[inline]
    fn add(self, rhs: Quaternion<T>) -> Self::Output {
        Quaternion {
            v: self.v + rhs.v,
            w: self.w + rhs.w,
        }
    }
}

impl<T: AddAssign> AddAssign<Quaternion<T>> for Quaternion<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Quaternion<T>) {
        self.v.add_assign(rhs.v);
        self.w.add_assign(rhs.w);
    }
}

impl<T: Sub> Sub<Quaternion<T>> for Quaternion<T> {
    type Output = Quaternion<T::Output>;
    #[inline]
    fn sub(self, rhs: Quaternion<T>) -> Self::Output {
        Quaternion {
            v: self.v - rhs.v,
            w: self.w - rhs.w,
        }
    }
}

impl<T: SubAssign> SubAssign<Quaternion<T>> for Quaternion<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Quaternion<T>) {
        self.v.sub_assign(rhs.v);
        self.w.sub_assign(rhs.w);
    }
}

impl<T: Copy + Add> Add<T> for Quaternion<T> {
    type Output = Quaternion<T::Output>;
    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Quaternion {
            v: self.v + Vector::splat(rhs),
            w: self.w + rhs,
        }
    }
}

impl<T: Copy + AddAssign> AddAssign<T> for Quaternion<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.v.add_assign(Vector::splat(rhs));
        self.w.add_assign(rhs);
    }
}

impl<T: Copy + Sub> Sub<T> for Quaternion<T> {
    type Output = Quaternion<T::Output>;
    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Quaternion {
            v: self.v - Vector::splat(rhs),
            w: self.w - rhs,
        }
    }
}

impl<T: Copy + SubAssign> SubAssign<T> for Quaternion<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.v.sub_assign(Vector::splat(rhs));
        self.w.sub_assign(rhs);
    }
}

impl<T: Copy + Mul> Mul<T> for Quaternion<T> {
    type Output = Quaternion<T::Output>;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Quaternion {
            v: self.v * rhs,
            w: self.w * rhs,
        }
    }
}

impl<T: Copy + CheckedMul> CheckedMul<T> for Quaternion<T> {
    #[inline]
    fn checked_mul(self, rhs: T) -> Option<Self::Output> {
        Some(Quaternion {
            v: self.v.checked_mul(rhs)?,
            w: self.w.checked_mul(rhs)?,
        })
    }
}

impl<T: Copy + MulAssign> MulAssign<T> for Quaternion<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.v.mul_assign(rhs);
        self.w.mul_assign(rhs);
    }
}

impl<T: Copy + Div> Div<T> for Quaternion<T> {
    type Output = Quaternion<T::Output>;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Quaternion {
            v: self.v / rhs,
            w: self.w / rhs,
        }
    }
}

impl<T: Copy + CheckedDiv> CheckedDiv<T> for Quaternion<T> {
    #[inline]
    fn checked_div(self, rhs: T) -> Option<Self::Output> {
        Some(Quaternion {
            v: self.v.checked_div(rhs)?,
            w: self.w.checked_div(rhs)?,
        })
    }
}

impl<T: Copy + DivAssign> DivAssign<T> for Quaternion<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.v.div_assign(rhs);
        self.w.div_assign(rhs);
    }
}

impl<T: Zero + One> Quaternion<T> {
    #[must_use]
    #[inline]
    pub const fn identity() -> Self {
        Self {
            v: Zero::ZERO,
            w: One::ONE,
        }
    }
}

impl<T: Zero + One> Default for Quaternion<T> {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

impl<T> From<Matrix4<T>> for Quaternion<T>
where
    T: ClosedAdd + Copy + ClosedDiv + ClosedMul + ClosedSub + One + PartialOrd + Zero + Sqrt,
{
    #[inline]
    fn from(value: Matrix4<T>) -> Self {
        Self::from_matrix(value)
    }
}

impl<T> From<Vector4<T>> for Quaternion<T> {
    #[inline]
    fn from(value: Vector4<T>) -> Self {
        Self::from_vector(value)
    }
}

impl<V: Into<Vector3<T>>, T> From<(V, T)> for Quaternion<T> {
    #[inline]
    fn from((vector, scalar): (V, T)) -> Self {
        Self::from_components(vector, scalar)
    }
}

impl<T> From<Quaternion<T>> for (Vector3<T>, T) {
    #[inline]
    fn from(value: Quaternion<T>) -> Self {
        (value.v, value.w)
    }
}

#[cfg(feature = "mint")]
impl<T> From<mint::Quaternion<T>> for Quaternion<T> {
    #[inline]
    fn from(value: mint::Quaternion<T>) -> Self {
        Quaternion {
            v: From::from(value.v),
            w: value.s,
        }
    }
}

#[cfg(feature = "mint")]
impl<T> From<Quaternion<T>> for mint::Quaternion<T> {
    #[inline]
    fn from(value: Quaternion<T>) -> Self {
        mint::Quaternion {
            v: From::from(value.v),
            s: value.w,
        }
    }
}

#[cfg(feature = "mint")]
impl<T> mint::IntoMint for Quaternion<T> {
    type MintType = mint::Quaternion<T>;
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Zeroable> bytemuck::Zeroable for Quaternion<T> {
    #[inline]
    fn zeroed() -> Self {
        Quaternion {
            v: bytemuck::Zeroable::zeroed(),
            w: bytemuck::Zeroable::zeroed(),
        }
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T: bytemuck::Pod> bytemuck::Pod for Quaternion<T> {}

#[cfg(feature = "approx")]
impl<T: approx::AbsDiffEq> approx::AbsDiffEq for Quaternion<T>
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
        self.v.abs_diff_eq(&other.v, epsilon) && self.w.abs_diff_eq(&other.w, epsilon)
    }
}

#[cfg(feature = "approx")]
impl<T: approx::RelativeEq> approx::RelativeEq for Quaternion<T>
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
        self.v.relative_eq(&other.v, epsilon, max_relative)
            && self.w.relative_eq(&other.w, epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl<T: approx::UlpsEq> approx::UlpsEq for Quaternion<T>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.v.ulps_eq(&other.v, epsilon, max_ulps) && self.w.ulps_eq(&other.w, epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use crate::rotation::quaternion::Quaternion;
    #[cfg(any(feature = "std", feature = "libm"))]
    use crate::{
        matrix::{Matrix, Matrix4},
        rotation::{Rotation, angle::Angle},
        vector::Vector,
    };
    use core::{
        ops::Neg,
        sync::atomic::{AtomicU32, Ordering},
    };

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_quat_multiply() {
        let q1 = Quaternion::from_components([1.0, 2.0, 4.0], 3.0);
        let q2 = Quaternion::identity();

        assert_eq!(q1 * q2, q1);
        assert_eq!(q2 * q1, q1);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_conjugate() {
        let q1 = Quaternion::from_angle_axis(Angle::<f64>::three_quarters(), Vector::X);
        let q2 = Quaternion::from_angle_axis(Angle::<f64>::three_quarters().neg(), Vector::X);

        assert!(q1.conjugated().is_nearly_equal(q2, 1e-14));

        let m1 = q1.into_homogeneous();
        let m2 = q1.conjugated().into_homogeneous();

        approx::assert_relative_eq!(m1 * m2, Matrix::<f64>::identity());
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[test]
    fn test_quat_from_matrix() {
        let mat = Matrix4::<f64>::identity();
        let quat = Quaternion::from_matrix(mat);

        assert_eq!(quat, Quaternion::identity());
        assert_eq!(Matrix4::rotation_3d(quat), mat);

        let mat = Matrix4::<f64>::x_axis_rotation(Angle::half());
        let quat = Quaternion::from_matrix(mat);

        approx::assert_relative_eq!(Matrix4::rotation_3d(quat), mat, epsilon = 1e-15);

        let axis = Vector::new([0.7, 0.3, 1.2]).normalized();
        let angle = Angle::Degrees(45.0);

        let mat = Matrix4::<f64>::axis_rotation_3d(angle, axis);
        let quat = Quaternion::from_angle_axis(angle, axis);

        approx::assert_relative_eq!(Matrix4::rotation_3d(quat), mat, epsilon = 1e-15);
    }

    #[test]
    fn test_conjugate_complex_type() {
        static NUM_INSTANCES: AtomicU32 = AtomicU32::new(0);

        #[derive(PartialEq, Eq, Hash, Debug)]
        struct SomeType(String);

        impl SomeType {
            fn new<S: Into<String>>(value: S) -> Self {
                NUM_INSTANCES.fetch_add(1, Ordering::SeqCst);
                Self(value.into())
            }
        }

        impl Clone for SomeType {
            #[inline]
            fn clone(&self) -> Self {
                NUM_INSTANCES.fetch_add(1, Ordering::SeqCst);
                Self(self.0.clone())
            }
        }

        impl Drop for SomeType {
            fn drop(&mut self) {
                NUM_INSTANCES.fetch_sub(1, Ordering::SeqCst);
            }
        }

        impl Neg for SomeType {
            type Output = Self;
            fn neg(self) -> Self::Output {
                self
            }
        }

        let quat = Quaternion::from_components(
            [
                SomeType::new("hello 1".to_string()),
                SomeType::new("hello 2".to_string()),
                SomeType::new("hello 3".to_string()),
            ],
            SomeType::new("hello 4".to_string()),
        );

        let conj = quat.clone().conjugated();
        assert_eq!(conj, quat);
        drop(conj);
        drop(quat);
        assert_eq!(NUM_INSTANCES.load(Ordering::SeqCst), 0);
    }

    #[cfg(any(feature = "std", feature = "libm"))]
    #[ignore = "I don't know how to math"]
    #[test]
    fn test_convert() {
        let (angle, axis) = (
            Angle::<f64>::quarter(),
            Vector::new([1.0, 5.0, 3.0]).normalized(),
        );

        let quat = Quaternion::from_angle_axis(angle, axis);

        let (angle_2, axis_2) = quat.into_angle_axis();
        assert_eq!(angle, angle_2);
        assert_eq!(axis, axis_2);
    }
}
