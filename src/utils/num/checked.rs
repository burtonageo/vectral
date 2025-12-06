use core::{
    error::Error,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign},
};

#[non_exhaustive]
#[derive(Debug)]
pub enum CheckedIntegerArithmeticError {
    Overflow,
    Underflow,
    DivideByZero,
}

impl fmt::Display for CheckedIntegerArithmeticError {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Overflow => fmtr.write_str("arithmetic operation would overflow"),
            Self::Underflow => fmtr.write_str("arithmetic operation would underflow"),
            Self::DivideByZero => fmtr.write_str("attempted divide by zero"),
        }
    }
}

impl Error for CheckedIntegerArithmeticError {}

#[non_exhaustive]
#[derive(Debug)]
pub enum CheckedFloatingPointArithmeticError {}

pub trait CheckedAdd<T = Self>: Add<T> {
    #[must_use]
    fn checked_add(self, rhs: T) -> Option<Self::Output>;
}

pub trait CheckedAddAssign<T = Self>: AddAssign<T> {
    type Error: fmt::Debug;
    fn checked_add_assign(&mut self, rhs: T) -> Result<(), Self::Error>;
}

pub trait CheckedSub<T = Self>: Sub<T> {
    #[must_use]
    fn checked_sub(self, rhs: T) -> Option<Self::Output>;
}

pub trait CheckedSubAssign<T = Self>: SubAssign<T> {
    type Error: fmt::Debug;
    fn checked_sub_assign(&mut self, rhs: T) -> Result<(), Self::Error>;
}

pub trait CheckedMul<T = Self>: Mul<T> {
    #[must_use]
    fn checked_mul(self, rhs: T) -> Option<Self::Output>;
}

pub trait CheckedMulAssign<T = Self>: MulAssign<T> {
    type Error: fmt::Debug;
    fn checked_mul_assign(&mut self, rhs: T) -> Result<(), Self::Error>;
}

pub trait CheckedDiv<T = Self>: Div<T> {
    #[must_use]
    fn checked_div(self, rhs: T) -> Option<Self::Output>;
}

pub trait CheckedDivAssign<T = Self>: DivAssign<T> {
    type Error: fmt::Debug;
    fn checked_div_assign(&mut self, rhs: T) -> Result<(), Self::Error>;
}

pub trait CheckedRem<T = Self>: Rem<T> {
    #[must_use]
    fn checked_rem(self, rhs: T) -> Option<Self::Output>;
}

pub trait CheckedRemAssign<T = Self>: RemAssign<T> {
    type Error: fmt::Debug;
    fn checked_rem_assign(&mut self, rhs: T) -> Result<(), Self::Error>;
}

macro_rules! impl_checked_int_ops {
    (
        $( $( #[ $meta:meta ] )* $int_ty:ty ),* $(,)?
    ) => {
        $(
            $( #[ $meta ] )*
            impl CheckedAdd for $int_ty {
                #[inline]
                fn checked_add(self, rhs: Self) -> Option<Self::Output> {
                    <$int_ty>::checked_add(self, rhs)
                }
            }

            $( #[ $meta ] )*
            impl CheckedSub for $int_ty {
                #[inline]
                fn checked_sub(self, rhs: Self) -> Option<Self::Output> {
                    <$int_ty>::checked_sub(self, rhs)
                }
            }

            $( #[ $meta ] )*
            impl CheckedMul for $int_ty {
                #[inline]
                fn checked_mul(self, rhs: Self) -> Option<Self::Output> {
                    <$int_ty>::checked_mul(self, rhs)
                }
            }

            $( #[ $meta ] )*
            impl CheckedDiv for $int_ty {
                #[inline]
                fn checked_div(self, rhs: Self) -> Option<Self::Output> {
                    <$int_ty>::checked_div(self, rhs)
                }
            }

            $( #[ $meta ] )*
            impl CheckedRem for $int_ty {
                #[inline]
                fn checked_rem(self, rhs: Self) -> Option<Self::Output> {
                    <$int_ty>::checked_rem(self, rhs)
                }
            }

            $( #[ $meta ] )*
            impl CheckedAddAssign for $int_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_add_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_add(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedSubAssign for $int_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_sub_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_sub(rhs).ok_or(CheckedIntegerArithmeticError::Underflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedMulAssign for $int_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_mul_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_mul(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedDivAssign for $int_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_div_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    if rhs == 0 {
                        return Err(CheckedIntegerArithmeticError::DivideByZero);
                    }

                    *self = self.checked_div(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedRemAssign for $int_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_rem_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_rem(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }
        )*
    };
}

impl_checked_int_ops! {
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
}

macro_rules! impl_checked_float_ops {
    (
        $( $( #[ $meta:meta ] )* $float_ty:ty ),* $(,)?
    ) => {
        $(
            $( #[ $meta ] )*
            impl CheckedAdd for $float_ty {
                #[inline]
                fn checked_add(self, rhs: Self) -> Option<Self::Output> {
                    let result = self + rhs;
                    if result < <$float_ty>::MAX || <$float_ty>::MAX - result == <$float_ty>::MAX - self - rhs {
                        Some(result)
                    } else {
                        None
                    }
                }
            }

            $( #[ $meta ] )*
            impl CheckedSub for $float_ty {
                #[inline]
                fn checked_sub(self, rhs: Self) -> Option<Self::Output> {
                    let result = self - rhs;
                    if result > <$float_ty>::MIN || <$float_ty>::MIN + result == <$float_ty>::MIN + self + rhs {
                        Some(result)
                    } else {
                        None
                    }
                }
            }

            $( #[ $meta ] )*
            impl CheckedMul for $float_ty {
                #[inline]
                fn checked_mul(self, rhs: Self) -> Option<Self::Output> {
                    let result = self * rhs;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
            }

            $( #[ $meta ] )*
            impl CheckedDiv for $float_ty {
                #[inline]
                fn checked_div(self, rhs: Self) -> Option<Self::Output> {
                    let result = self / rhs;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
            }

            $( #[ $meta ] )*
            impl CheckedRem for $float_ty {
                #[inline]
                fn checked_rem(self, rhs: Self) -> Option<Self::Output> {
                    let result = self % rhs;
                    if result.is_finite() {
                        Some(result)
                    } else {
                        None
                    }
                }
            }

            $( #[ $meta ] )*
            impl CheckedAddAssign for $float_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_add_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_add(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedSubAssign for $float_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_sub_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_sub(rhs).ok_or(CheckedIntegerArithmeticError::Underflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedMulAssign for $float_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_mul_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_mul(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedDivAssign for $float_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_div_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    if rhs == 0.0 {
                        return Err(CheckedIntegerArithmeticError::DivideByZero);
                    }

                    *self = self.checked_div(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }

            $( #[ $meta ] )*
            impl CheckedRemAssign for $float_ty {
                type Error = CheckedIntegerArithmeticError;
                #[inline]
                fn checked_rem_assign(&mut self, rhs: Self) -> Result<(), Self::Error> {
                    *self = self.checked_rem(rhs).ok_or(CheckedIntegerArithmeticError::Overflow)?;
                    Ok(())
                }
            }
        )*
    };
}

impl_checked_float_ops! {
    f16, f32, f64, f128,
}
