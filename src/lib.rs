//! Wrapping-in-range arithmetic for custom ranges via the [`WrappingInRange`] type
//!
//! These arithmetic operations act just like `std`'s `.wrapping_sub()`, `.wrapping_add()`, etc. but for a custom user-provided range.
//!
//! # Examples
//!
//! ```
//! use wrapping_in_range::WrappingInRange;
//!
//! let w = |i: i16| WrappingInRange(i, -1..=1);
//!
//! assert_eq!(
//!     [-2, -1, 0, 1, 2].map(|i| w(i) - 1),
//!     [ 0, 1, -1, 0, 1]
//! );
//! assert_eq!(
//!     [-2, -1, 0,  1, 2].map(|i| w(i) + 1),
//!     [-1,  0, 1, -1, 0]
//! );
//! ```
#![cfg_attr(not(test), no_std)]

use core::{
    hash::Hash,
    ops::{
        Add, AddAssign, Bound, Div, DivAssign, Mul, MulAssign, Neg, Range, RangeBounds, Rem,
        RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
};
use num_traits::{
    Bounded, Euclid, NumOps, One, WrappingAdd, WrappingMul, WrappingNeg, WrappingShl, WrappingShr,
    WrappingSub,
};

/// This type has overloaded operators that allow wrapping-in-range arithmetic
///
/// # Examples
///
/// ```
/// use wrapping_in_range::WrappingInRange;
///
/// let mut x = WrappingInRange(0, -1..=1);
///
/// x += 1;
/// assert_eq!(x.0, 1);
///
/// x += 1;
/// assert_eq!(x.0, -1);
///
/// // Wrapped
/// x += 1;
/// assert_eq!(x.0, 0);
///
/// // Wrapped
/// x += 2;
/// assert_eq!(x.0, -1);
///
/// // Wrap back around
/// x -= 1;
/// assert_eq!(x.0, 1);
/// ```
///
/// # Trait implementations
///
/// The traits [`Eq`], [`Ord`], [`PartialEq`], [`PartialOrd`], [`Hash`] are implementing by delegating to the
/// first element, completely ignoring the range.
///
/// ```
/// use wrapping_in_range::WrappingInRange;
///
/// let w = WrappingInRange(4, -10_i8..=10);
///
/// assert!(w < 8);
/// assert!(w > WrappingInRange(2, -20..20));
/// assert_eq!(w, 4);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct WrappingInRange<N, B>(pub N, pub B);

impl<N, B> PartialEq<N> for WrappingInRange<N, B>
where
    N: PartialEq,
{
    fn eq(&self, other: &N) -> bool {
        self.0.eq(other)
    }
}

impl<N, B> PartialOrd<N> for WrappingInRange<N, B>
where
    N: PartialOrd,
{
    fn partial_cmp(&self, other: &N) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl<N, B, K> PartialEq<WrappingInRange<N, K>> for WrappingInRange<N, B>
where
    N: PartialEq,
{
    fn eq(&self, other: &WrappingInRange<N, K>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<N, B, K> PartialOrd<WrappingInRange<N, K>> for WrappingInRange<N, B>
where
    N: PartialOrd,
{
    fn partial_cmp(&self, other: &WrappingInRange<N, K>) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<N: Hash, B> Hash for WrappingInRange<N, B> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<N, B> Add<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn add(self, rhs: N) -> N {
        wrap_in_range(self.0.wrapping_add(&rhs), self.1)
    }
}

impl<N, B> Sub<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn sub(self, rhs: N) -> N {
        wrap_in_range(self.0.wrapping_sub(&rhs), self.1)
    }
}

impl<N, B> Mul<N> for WrappingInRange<N, B>
where
    N: Wrappable + WrappingMul,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn mul(self, rhs: N) -> N {
        wrap_in_range(self.0.wrapping_mul(&rhs), self.1)
    }
}

impl<N, B> Div<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn div(self, rhs: N) -> N {
        wrap_in_range(self.0 / rhs, self.1)
    }
}

impl<N, B> Rem<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn rem(self, rhs: N) -> N {
        wrap_in_range(self.0.rem_euclid(&rhs), self.1)
    }
}

impl<N, B> Neg for WrappingInRange<N, B>
where
    N: Wrappable + WrappingNeg,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn neg(self) -> N {
        wrap_in_range(self.0.wrapping_neg(), self.1)
    }
}

impl<N, B> Shl<u32> for WrappingInRange<N, B>
where
    N: Wrappable + WrappingShl,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn shl(self, rhs: u32) -> N {
        wrap_in_range(self.0.wrapping_shl(rhs), self.1)
    }
}

impl<N, B> Shr<u32> for WrappingInRange<N, B>
where
    N: Wrappable + WrappingShr,
    B: RangeBounds<N>,
{
    type Output = N;

    #[inline(always)]
    fn shr(self, rhs: u32) -> N {
        wrap_in_range(self.0.wrapping_shr(rhs), self.1)
    }
}

impl<N, B> AddAssign<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N> + Clone,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: N) {
        self.0 = self.0.wrapping_add(&rhs);
        self.0 = wrap_in_range(self.0, self.1.clone());
    }
}

impl<N, B> SubAssign<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N> + Clone,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: N) {
        self.0 = self.0.wrapping_sub(&rhs);
        self.0 = wrap_in_range(self.0, self.1.clone());
    }
}

impl<N, B> MulAssign<N> for WrappingInRange<N, B>
where
    N: Wrappable + WrappingMul,
    B: RangeBounds<N> + Clone,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: N) {
        self.0 = self.0.wrapping_mul(&rhs);
        self.0 = wrap_in_range(self.0, self.1.clone());
    }
}

impl<N, B> DivAssign<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N> + Clone,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: N) {
        self.0 = self.0 / rhs;
        self.0 = wrap_in_range(self.0, self.1.clone());
    }
}

impl<N, B> RemAssign<N> for WrappingInRange<N, B>
where
    N: Wrappable,
    B: RangeBounds<N> + Clone,
{
    #[inline(always)]
    fn rem_assign(&mut self, rhs: N) {
        self.0 = self.0.rem_euclid(&rhs);
        self.0 = wrap_in_range(self.0, self.1.clone());
    }
}

impl<N, B> ShlAssign<u32> for WrappingInRange<N, B>
where
    N: Wrappable + WrappingShl,
    B: RangeBounds<N> + Clone,
{
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0 = self.0.wrapping_shl(rhs);
        self.0 = wrap_in_range(self.0, self.1.clone());
    }
}

impl<N, B> ShrAssign<u32> for WrappingInRange<N, B>
where
    N: Wrappable + WrappingShr,
    B: RangeBounds<N> + Clone,
{
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0 = self.0.wrapping_shr(rhs);
        self.0 = wrap_in_range(self.0, self.1.clone());
    }
}

/// Trait alias for types that can be wrapped inside of a range
trait Wrappable: NumOps + Copy + One + Bounded + WrappingAdd + WrappingSub + Euclid {}
impl<N: NumOps + Copy + One + Bounded + WrappingAdd + WrappingSub + Euclid> Wrappable for N {}

/// Converts `impl RangeBounds<N>` into `Range<N>`
fn norm_range<N: Wrappable, B: RangeBounds<N>>(range: B) -> Range<N> {
    // Minimum value the range can be
    let min = match range.start_bound() {
        Bound::Included(n) => *n,
        Bound::Excluded(n) => n.wrapping_add(&N::one()),
        Bound::Unbounded => N::min_value(),
    };

    // Maximum value the range can be
    let max = match range.end_bound() {
        Bound::Included(n) => *n,
        Bound::Excluded(n) => n.wrapping_sub(&N::one()),
        Bound::Unbounded => N::max_value(),
    };

    // How many integers there are in the range
    let range_size = max.wrapping_sub(&min).wrapping_add(&N::one());
    min..min.wrapping_add(&range_size)
}

/// Wrap `n` so that it fits inside of `range`
#[inline]
fn wrap_in_range<N, R>(n: N, range: R) -> N
where
    N: Wrappable,
    R: RangeBounds<N>,
{
    let Range { start, end } = norm_range(range);
    let range_size = end.wrapping_sub(&start);
    let wrapped_val = n.wrapping_sub(&start).rem_euclid(&range_size);
    wrapped_val.wrapping_add(&start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::ops::RangeInclusive;

    /// Our test range
    const RANGE: RangeInclusive<i16> = -100i16..=100i16;

    /// Number that could be outside of the range
    fn outside_of_range() -> impl Strategy<Value = i16> {
        any::<i16>()
    }

    /// Numbers in tests are manually wrapped
    fn manual_wrap(n: i16, range: RangeInclusive<i16>) -> i16 {
        let (start, end) = range.into_inner();
        let range_size = end.wrapping_sub(start).wrapping_add(1);
        let mut val = n.wrapping_sub(start);
        val = val.rem_euclid(range_size);
        val.wrapping_add(start)
    }

    #[test]
    fn cmp_ord() {
        let w = WrappingInRange(1, 10..40);
        assert!(w > 0);
        assert!(w < 2);
        assert!(w == 1);

        let w2 = WrappingInRange(1, 12..=30);
        assert!(w >= w2);
        assert!(w == w2);
    }

    proptest! {
        #[test]
        fn add(n in RANGE, x in outside_of_range()) {
            let wrapped_result = WrappingInRange(n, RANGE) + x;
            let std_result = n.wrapping_add(x);
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn sub(n in RANGE, x in outside_of_range()) {
            let wrapped_result = WrappingInRange(n, RANGE) - x;
            let std_result = n.wrapping_sub(x);
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn mul(n in RANGE, x in outside_of_range()) {
            let wrapped_result = WrappingInRange(n, RANGE) * x;
            let std_result = n.wrapping_mul(x);
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn div(n in outside_of_range(), x in outside_of_range().prop_filter("non-zero", |&x| x != 0)) {
            let wrapped_result = WrappingInRange(n, RANGE) / x;
            let std_result = n / x;
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn rem(n in outside_of_range(), x in outside_of_range().prop_filter("non-zero", |&x| x != 0)) {
            let wrapped_result = WrappingInRange(n, RANGE) % x;
            let std_result = n.rem_euclid(x);
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn neg(n in RANGE) {
            let wrapped_result = -WrappingInRange(n, RANGE);
            let std_result = n.wrapping_neg();
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn shl(n in RANGE, x in 0u32..=15) {
            let wrapped_result = WrappingInRange(n, RANGE) << x;
            let std_result = n.wrapping_shl(x);
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn shr(n in RANGE, x in 0u32..=15) {
            let wrapped_result = WrappingInRange(n, RANGE) >> x;
            let std_result = n.wrapping_shr(x);
            prop_assert_eq!(
                wrapped_result,
                manual_wrap(std_result, RANGE)
            );
        }

        #[test]
        fn add_assign(n in RANGE, x in outside_of_range()) {
            let mut val = WrappingInRange(n, RANGE);
            let mut manual_val = n;
            val += x;
            manual_val = manual_wrap(manual_val.wrapping_add(x), RANGE);
            prop_assert_eq!(val.0, manual_val);
        }

        #[test]
        fn sub_assign(n in RANGE, x in outside_of_range()) {
            let mut val = WrappingInRange(n, RANGE);
            let mut manual_val = n;
            val -= x;
            manual_val = manual_wrap(manual_val.wrapping_sub(x), RANGE);
            prop_assert_eq!(val.0, manual_val);
        }

        #[test]
        fn mul_assign(n in RANGE, x in outside_of_range()) {
            let mut val = WrappingInRange(n, RANGE);
            let mut manual_val = n;
            val *= x;
            manual_val = manual_wrap(manual_val.wrapping_mul(x), RANGE);
            prop_assert_eq!(val.0, manual_val);
        }

        #[test]
        fn div_assign(n in outside_of_range(), x in outside_of_range().prop_filter("non-zero", |&x| x != 0)) {
            let mut val = WrappingInRange(n, RANGE);
            let mut manual_val = n;
            val /= x;
            manual_val = manual_wrap(manual_val / x, RANGE);
            prop_assert_eq!(val.0, manual_val);
        }

        #[test]
        fn rem_assign(n in outside_of_range(), x in outside_of_range().prop_filter("non-zero", |&x| x != 0)) {
            let mut val = WrappingInRange(n, RANGE);
            let mut manual_val = n;
            val %= x;
            manual_val = manual_wrap(manual_val.rem_euclid(x), RANGE);
            prop_assert_eq!(val.0, manual_val);
        }

        #[test]
        fn shl_assign(n in RANGE, x in 0u32..=15) {
            let mut val = WrappingInRange(n, RANGE);
            let mut manual_val = n;
            val <<= x;
            manual_val = manual_wrap(manual_val.wrapping_shl(x), RANGE);
            prop_assert_eq!(val.0, manual_val);
        }

        #[test]
        fn shr_assign(n in RANGE, x in 0u32..=15) {
            let mut val = WrappingInRange(n, RANGE);
            let mut manual_val = n;
            val >>= x;
            manual_val = manual_wrap(manual_val.wrapping_shr(x), RANGE);
            prop_assert_eq!(val.0, manual_val);
        }
    }
}
