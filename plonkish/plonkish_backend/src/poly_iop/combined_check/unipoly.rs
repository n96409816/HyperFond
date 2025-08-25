#![allow(dead_code)]
use std::{
    cmp::Ordering,
    ops::{AddAssign, Index, IndexMut, Mul, MulAssign},
};

use crate::{
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, BooleanHypercube, PrimeField},
        end_timer,
        expression::Expression,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use super::gaussian_elimination::gaussian_elimination;

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniPoly<F: PrimeField> {
    pub coeffs: Vec<F>,
}

// ax^2 + bx + c stored as vec![c,a]
// ax^3 + bx^2 + cx + d stored as vec![d,b,a]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompressedUniPoly<F: PrimeField> {
    coeffs_except_linear_term: Vec<F>,
}

impl<F: PrimeField> UniPoly<F> {
    #[allow(dead_code)]
    pub fn from_coeff(coeffs: Vec<F>) -> Self {
        UniPoly { coeffs }
    }

    pub fn from_evals(evals: &[F]) -> Self {
        UniPoly {
            coeffs: Self::vandermonde_interpolation(evals),
        }
    }

    fn vandermonde_interpolation(evals: &[F]) -> Vec<F> {
        let n = evals.len();
        let xs: Vec<F> = (0..n).map(|x| F::from(x as u64)).collect();

        let mut vandermonde: Vec<Vec<F>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            let x = xs[i];
            row.push(F::ONE);
            row.push(x);
            for j in 2..n {
                row.push(row[j - 1] * x);
            }
            row.push(evals[i]);
            vandermonde.push(row);
        }

        gaussian_elimination(&mut vandermonde)
    }

    /// Divide self by another polynomial, and returns the
    /// quotient and remainder.
    pub fn divide_with_remainder(&self, divisor: &Self) -> Option<(Self, Self)> {
        if self.is_zero() {
            Some((Self::zero(), Self::zero()))
        } else if divisor.is_zero() {
            None
        } else if self.degree() < divisor.degree() {
            Some((Self::zero(), self.clone()))
        } else {
            // Now we know that self.degree() >= divisor.degree();
            let mut quotient = vec![F::ZERO; self.degree() - divisor.degree() + 1];
            let mut remainder: Self = self.clone();
            // Can unwrap here because we know self is not zero.
            let divisor_leading_inv = divisor.leading_coefficient().unwrap().invert().unwrap();
            while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
                let cur_q_coeff = *remainder.leading_coefficient().unwrap() * divisor_leading_inv;
                let cur_q_degree = remainder.degree() - divisor.degree();
                quotient[cur_q_degree] = cur_q_coeff;

                for (i, div_coeff) in divisor.coeffs.iter().enumerate() {
                    remainder.coeffs[cur_q_degree + i] -= cur_q_coeff * *div_coeff;
                }
                while let Some(true) = remainder.coeffs.last().map(|c| c == &F::ZERO) {
                    remainder.coeffs.pop();
                }
            }
            Some((Self::from_coeff(quotient), remainder))
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|c| c == &F::ZERO)
    }

    fn leading_coefficient(&self) -> Option<&F> {
        self.coeffs.last()
    }

    fn zero() -> Self {
        Self::from_coeff(Vec::new())
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn as_vec(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    pub fn eval_at_zero(&self) -> F {
        self.coeffs[0]
    }

    pub fn eval_at_one(&self) -> F {
        (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
    }

    pub fn evaluate(&self, r: &F) -> F {
        let mut eval = self.coeffs[0];
        let mut power = *r;
        for i in 1..self.coeffs.len() {
            eval += power * self.coeffs[i];
            power *= *r;
        }
        eval
    }

    pub fn compress(&self) -> CompressedUniPoly<F> {
        let coeffs_except_linear_term = [&self.coeffs[..1], &self.coeffs[2..]].concat();
        debug_assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
        CompressedUniPoly {
            coeffs_except_linear_term,
        }
    }

    pub fn shift_coefficients(&mut self, rhs: &F) {
        self.coeffs.par_iter_mut().for_each(|c| *c += *rhs);
    }
}

// impl<F: PrimeField> AddAssign<&F> for UniPoly<F> {
// fn add_assign(&mut self, rhs: &F) {
// self.coeffs.par_iter_mut().for_each(|c| *c += rhs);
// }
// }

impl<F: PrimeField> AddAssign<&Self> for UniPoly<F> {
    fn add_assign(&mut self, rhs: &Self) {
        let ordering = self.coeffs.len().cmp(&rhs.coeffs.len());
        #[allow(clippy::disallowed_methods)]
        for (lhs, rhs) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
            *lhs += *rhs;
        }
        if matches!(ordering, Ordering::Less) {
            self.coeffs
                .extend(rhs.coeffs[self.coeffs.len()..].iter().cloned());
        }
    }
}

impl<F: PrimeField> Mul<F> for UniPoly<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self {
        let iter = self.coeffs.into_par_iter();
        Self::from_coeff(iter.map(|c| c * rhs).collect::<Vec<_>>())
    }
}

impl<F: PrimeField> Mul<&F> for UniPoly<F> {
    type Output = Self;

    fn mul(self, rhs: &F) -> Self {
        let iter = self.coeffs.into_par_iter();
        Self::from_coeff(iter.map(|c| c * *rhs).collect::<Vec<_>>())
    }
}

impl<F: PrimeField> Index<usize> for UniPoly<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coeffs[index]
    }
}

impl<F: PrimeField> IndexMut<usize> for UniPoly<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coeffs[index]
    }
}

impl<F: PrimeField> MulAssign<&F> for UniPoly<F> {
    fn mul_assign(&mut self, rhs: &F) {
        self.coeffs.par_iter_mut().for_each(|c| *c *= *rhs);
    }
}

impl<F: PrimeField> CompressedUniPoly<F> {
    // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
    // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
    pub fn decompress(&self, hint: &F) -> UniPoly<F> {
        let mut linear_term =
            *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term -= self.coeffs_except_linear_term[i];
        }

        let mut coeffs = vec![self.coeffs_except_linear_term[0], linear_term];
        coeffs.extend(&self.coeffs_except_linear_term[1..]);
        assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
        UniPoly { coeffs }
    }

    pub fn decompress_zerocheck(&self, hint: &F, alpha_inv: &F) -> UniPoly<F> {
        let linear_term = *alpha_inv * (*hint - self.coeffs_except_linear_term[0])
            - self.coeffs_except_linear_term[1];

        let mut coeffs = vec![self.coeffs_except_linear_term[0], linear_term];
        coeffs.extend(&self.coeffs_except_linear_term[1..]);
        assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
        UniPoly { coeffs }
    }
}

// impl<F: PrimeField> AppendToTranscript for UniPoly<F> {
//     fn append_to_transcript(&self, label: &'static [u8], transcript: &mut
// ProofTranscript) {         transcript.append_message(label,
// b"UniPoly_begin");         for i in 0..self.coeffs.len() {
//             transcript.append_scalar(b"coeff", &self.coeffs[i]);
//         }
//         transcript.append_message(label, b"UniPoly_end");
//     }
// }
pub fn interpolate_uni_poly<F: PrimeField>(p_i: &[F], eval_at: F) -> F {
    let len = p_i.len();
    let mut evals = vec![];
    let mut prod = eval_at;
    evals.push(eval_at);

    // `prod = \prod_{j} (eval_at - j)`
    for e in 1..len {
        let tmp = eval_at - F::from(e as u64);
        evals.push(tmp);
        prod *= tmp;
    }
    let mut res = F::ZERO;
    // we want to compute \prod (j!=i) (i-j) for a given i
    //
    // we start from the last step, which is
    //  denom[len-1] = (len-1) * (len-2) *... * 2 * 1
    // the step before that is
    //  denom[len-2] = (len-2) * (len-3) * ... * 2 * 1 * -1
    // and the step before that is
    //  denom[len-3] = (len-3) * (len-4) * ... * 2 * 1 * -1 * -2
    //
    // i.e., for any i, the one before this will be derived from
    //  denom[i-1] = denom[i] * (len-i) / i
    //
    // that is, we only need to store
    // - the last denom for i = len-1, and
    // - the ratio between current step and fhe last step, which is the product of
    //   (len-i) / i from all previous steps and we store this product as a fraction
    //   number to reduce field divisions.

    // We know
    //  - 2^61 < factorial(20) < 2^62
    //  - 2^122 < factorial(33) < 2^123
    // so we will be able to compute the ratio
    //  - for len <= 20 with i64
    //  - for len <= 33 with i128
    //  - for len >  33 with BigInt
    if p_i.len() <= 20 {
        let last_denominator = F::from(u64_factorial(len - 1));
        let mut ratio_numerator = 1i64;
        let mut ratio_denominator = 1u64;

        for i in (0..len).rev() {
            let ratio_numerator_f = if ratio_numerator < 0 {
                -F::from((-ratio_numerator) as u64)
            } else {
                F::from(ratio_numerator as u64)
            };

            let inv = (last_denominator * ratio_numerator_f * evals[i])
                .invert()
                .unwrap();
            res += p_i[i] * prod * F::from(ratio_denominator) * inv;
            // / (last_denominator * ratio_numerator_f * evals[i]);

            // compute denom for the next step is current_denom * (len-i)/i
            if i != 0 {
                ratio_numerator *= -(len as i64 - i as i64);
                ratio_denominator *= i as u64;
            }
        }
    } else if p_i.len() <= 33 {
        let last_denominator = F::from(u128_factorial(len - 1).try_into().unwrap());
        let mut ratio_numerator = 1i128;
        let mut ratio_denominator = 1u128;

        for i in (0..len).rev() {
            let ratio_numerator_f = if ratio_numerator < 0 {
                -F::from(((-ratio_numerator) as u128).try_into().unwrap())
            } else {
                F::from((ratio_numerator as u128).try_into().unwrap())
            };

            let inv = (last_denominator * ratio_numerator_f * evals[i])
                .invert()
                .unwrap();
            res += p_i[i] * prod * F::from(ratio_denominator.try_into().unwrap()) * inv;

            // compute denom for the next step is current_denom * (len-i)/i
            if i != 0 {
                ratio_numerator *= -(len as i128 - i as i128);
                ratio_denominator *= i as u128;
            }
        }
    } else {
        let mut denom_up = field_factorial::<F>(len - 1);
        let mut denom_down = F::ONE;

        for i in (0..len).rev() {
            let inv = (denom_up * evals[i]).invert().unwrap();
            res += p_i[i] * prod * denom_down * inv;

            // compute denom for the next step is current_denom * (len-i)/i
            if i != 0 {
                denom_up *= -F::from((len - i) as u64);
                denom_down *= F::from(i as u64);
            }
        }
    }
    res
}

/// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn field_factorial<F: PrimeField>(a: usize) -> F {
    let mut res = F::ONE;
    for i in 2..=a {
        res *= F::from(i as u64);
    }
    res
}

/// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn u128_factorial(a: usize) -> u128 {
    let mut res = 1u128;
    for i in 2..=a {
        res *= i as u128;
    }
    res
}

/// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn u64_factorial(a: usize) -> u64 {
    let mut res = 1u64;
    for i in 2..=a {
        res *= i as u64;
    }
    res
}
