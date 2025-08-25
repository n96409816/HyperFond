// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use super::ArithErrors;
use crate::poly::multilinear::MultilinearPolynomial;
use crate::Math;
use ark_std::sync::Arc;
use ark_std::{end_timer, log2, start_timer};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet, Stats};
use ff::{BatchInverter, PrimeField};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
/// Decompose an integer into a binary vector in little endian.
pub fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
    let mut res = Vec::with_capacity(num_var);
    let mut i = input;
    for _ in 0..num_var {
        res.push(i & 1 == 1);
        i >>= 1;
    }
    res
}

pub fn bind_poly_var_bot_par<F: PrimeField>(
    poly: &mut MultilinearPolynomial<F>,
    r: &F,
    concurrency: usize,
) {
    let n = poly.evals.len() / 2;
    let mut chunk_size = (poly.evals.len() + concurrency - 1) / concurrency;
    if chunk_size == 0 || chunk_size % 2 == 1 {
        chunk_size += 1;
    }
    let num_chunks = (poly.evals.len() + chunk_size - 1) / chunk_size;
    poly.evals.par_chunks_mut(chunk_size).for_each(|chunk| {
        for i in 0..chunk.len() / 2 {
            chunk[i] = chunk[2 * i] + *r * (chunk[2 * i + 1] - chunk[2 * i]);
        }
    });
    for i in 1..num_chunks {
        let src_start = i * chunk_size;
        let dst_start = (i * chunk_size) / 2;
        let size = (std::cmp::min((i + 1) * chunk_size, poly.evals.len()) - src_start) / 2;
        unsafe {
            let data = poly.evals.as_mut_ptr();
            std::ptr::copy_nonoverlapping(data.add(src_start), data.add(dst_start), size);
        }
    }
    poly.num_vars -= 1;
    poly.evals.truncate(n);
}

pub fn build_eq_table<F: PrimeField>(r: &[F], coeff: F) -> Vec<Vec<F>> {
    let mut table: Vec<Vec<F>> = Vec::with_capacity(r.len());
    table.push(vec![coeff]);
    for r in r.iter().skip(1).rev() {
        let last = table.last().unwrap();
        let mut evals: Vec<F> = unsafe_allocate_zero_vec(last.len() * 2);
        evals
            .par_chunks_exact_mut(2)
            .zip(last.par_iter())
            .for_each(|(evals, last)| {
                evals[1] = *last * *r;
                evals[0] = *last - evals[1];
            });
        table.push(evals);
    }
    table.reverse();
    table
}

pub fn extrapolate<F: PrimeField>(points: &[F], weights: &[F], evals: &[F], at: &F) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *at - point).collect::<Vec<_>>();
        // batch_inversion(&mut coeffs);
        // let mut elements = values.clone();
        let mut scratch_space = vec![F::ZERO; coeffs.len()];
        BatchInverter::invert_with_external_scratch(&mut coeffs, &mut scratch_space);
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= weight;
        });
        let sum_inv = coeffs.iter().sum::<F>().invert().unwrap();
        (coeffs, sum_inv)
    };
    coeffs
        .iter()
        .zip(evals)
        .map(|(coeff, eval)| *coeff * eval)
        .sum::<F>()
        * sum_inv
}

/// given the evaluation input `point` of the `index`-th polynomial,
/// obtain the evaluation point in the merged polynomial
pub fn gen_eval_point<F: PrimeField>(index: usize, index_len: usize, point: &[F]) -> Vec<F> {
    let index_vec: Vec<F> = bit_decompose(index as u64, index_len)
        .into_iter()
        .map(|x| F::from(x as u64))
        .collect();
    [point, &index_vec].concat()
}

/// Return the number of variables that one need for an MLE to
/// batch the list of MLEs
#[inline]
pub fn get_batched_nv(num_var: usize, polynomials_len: usize) -> usize {
    num_var + log2(polynomials_len) as usize
}

// Input index
// - `i := (i_0, ...i_{n-1})`,
// - `num_vars := n`
// return three elements:
// - `x0 := (i_1, ..., i_{n-1}, 0)`
// - `x1 := (i_1, ..., i_{n-1}, 1)`
// - `sign := i_0`
#[inline]
pub fn get_index(i: usize, num_vars: usize) -> (usize, usize, bool) {
    let bit_sequence = bit_decompose(i as u64, num_vars);

    // the last bit comes first here because of LE encoding
    let x0 = project(&[[false].as_ref(), bit_sequence[..num_vars - 1].as_ref()].concat()) as usize;
    let x1 = project(&[[true].as_ref(), bit_sequence[..num_vars - 1].as_ref()].concat()) as usize;

    (x0, x1, bit_sequence[num_vars - 1])
}

/// Project a little endian binary vector into an integer.
#[inline]
pub(crate) fn project(input: &[bool]) -> u64 {
    let mut res = 0;
    for &e in input.iter().rev() {
        res <<= 1;
        res += e as u64;
    }
    res
}

pub fn unsafe_allocate_zero_vec<F: PrimeField + Sized>(size: usize) -> Vec<F> {
    // https://stackoverflow.com/questions/59314686/how-to-efficiently-create-a-large-vector-of-items-initialized-to-the-same-value

    // Check for safety of 0 allocation
    #[cfg(debug_assertions)]
    unsafe {
        let value = &F::ZERO;
        let ptr = value as *const F as *const u8;
        let bytes = std::slice::from_raw_parts(ptr, std::mem::size_of::<F>());
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    // Bulk allocate zeros, unsafely
    let result: Vec<F>;
    unsafe {
        let layout = std::alloc::Layout::array::<F>(size).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut F;

        if ptr.is_null() {
            panic!("Zero vec allocaiton failed");
        }

        result = Vec::from_raw_parts(ptr, size, size);
    }
    result
}

pub fn products_except_self<F: PrimeField>(x: &[F]) -> Vec<F> {
    let mut products = vec![F::ONE; x.len()];
    for i in 1..products.len() {
        products[i] = products[i - 1] * x[i - 1];
    }
    // (1, f_1, f_1 f_2, ... f_1f_2..f_{n-2})
    let mut running_suffix = F::ONE;
    for i in (0..(products.len() - 1)).rev() {
        running_suffix *= x[i + 1];
        products[i] *= running_suffix;
    }
    products
}

pub fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> Result<F, ArithErrors> {
    if x.len() != y.len() {
        return Err(ArithErrors::InvalidParameters(
            "x and y have different length".to_string(),
        ));
    }
    let start = start_timer!(|| "eq_eval");
    let mut res = F::ONE;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xi_yi = xi * yi;
        res *= xi_yi + xi_yi - xi - yi + F::ONE;
    }
    end_timer!(start);
    Ok(res)
}

/// Returns the evaluations of two list of MLEs:
/// - numerators = (a1, ..., ak)
/// - denominators = (b1, ..., bk)
///
///  where
///  - beta and gamma are challenges
///  - (f1, ..., fk), (g1, ..., gk),
///  - (s_id1, ..., s_idk), (perm1, ..., permk) are mle-s
///
/// - ai(x) is the MLE for `fi(x) + \beta s_id_i(x) + \gamma`
/// - bi(x) is the MLE for `gi(x) + \beta perm_i(x) + \gamma`
///
/// The caller is responsible for sanity-check
#[allow(clippy::type_complexity)]
pub(crate) fn compute_leaves<F: PrimeField, const DISTRIBUTED: bool>(
    beta: &F,
    gamma: &F,
    fxs: &[MultilinearPolynomial<F>],
    gxs: &[MultilinearPolynomial<F>],
    perms: &[MultilinearPolynomial<F>],
) -> Result<Vec<Vec<Vec<F>>>, ArithErrors> {
    let timer = start_timer!(|| "compute numerators and denominators");

    let mut leaves = vec![];

    let mut shift = 0;

    let n_parties = if DISTRIBUTED {
        Net::n_parties() as u64
    } else {
        1
    };
    let mut start = 0;
    while start < fxs.len() {
        let num_vars = fxs[start].num_vars;
        let mut end = start + 1;
        while end < fxs.len() && fxs[end].num_vars == num_vars {
            end += 1;
        }

        let (mut numerators, mut denominators) = (start..end)
            .into_par_iter()
            .map(|l| {
                let eval_len = num_vars.pow2() as u64;

                let start = if DISTRIBUTED {
                    shift
                        + eval_len * n_parties * ((l - start) as u64)
                        + eval_len * (Net::party_id() as u64)
                } else {
                    shift + eval_len * ((l - start) as u64)
                };

                (&fxs[l].evals, &gxs[l].evals, &perms[l].evals)
                    .into_par_iter()
                    .enumerate()
                    .map(|(i, (&f_ev, &g_ev, &perm_ev))| {
                        let numerator = f_ev + *beta * F::from(start + (i as u64)) + gamma;
                        let denominator = g_ev + *beta * perm_ev + gamma;
                        (numerator, denominator)
                    })
                    .unzip::<_, _, Vec<_>, Vec<_>>()
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();
        numerators.append(&mut denominators);
        leaves.push(numerators);

        shift += ((end - start) as u64) * (num_vars.pow2() as u64) * n_parties;
        start = end;
    }

    end_timer!(timer);
    Ok(leaves)
}

pub fn barycentric_weights<F: PrimeField + FnOnce() -> F>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter_map(|(i, point_i)| (i != j).then(|| *point_j - point_i))
                .reduce(|acc, value| acc * value)
                .unwrap_or_else(F::ONE)
        })
        .collect::<Vec<_>>();
    let mut scratch_space = vec![F::ZERO; weights.len()];
    BatchInverter::invert_with_external_scratch(&mut weights, &mut scratch_space);
    weights
}

#[cfg(test)]
mod test {
    use super::{bit_decompose, get_index, project};
    use ark_std::{rand::RngCore, test_rng};

    #[test]
    fn test_decomposition() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let t = rng.next_u64();
            let b = bit_decompose(t, 64);
            let r = project(&b);
            assert_eq!(t, r)
        }
    }

    #[test]
    fn test_get_index() {
        let a = 0b1010;
        let (x0, x1, sign) = get_index(a, 4);
        assert_eq!(x0, 0b0100);
        assert_eq!(x1, 0b0101);
        assert!(sign);

        let (x0, x1, sign) = get_index(a, 5);
        assert_eq!(x0, 0b10100);
        assert_eq!(x1, 0b10101);
        assert!(!sign);

        let a = 0b1111;
        let (x0, x1, sign) = get_index(a, 4);
        assert_eq!(x0, 0b1110);
        assert_eq!(x1, 0b1111);
        assert!(sign);
    }
}
