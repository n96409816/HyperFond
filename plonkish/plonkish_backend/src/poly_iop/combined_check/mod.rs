use self::{
    unipoly::{interpolate_uni_poly, UniPoly},
    util::{
        bind_poly_var_bot_par, bit_decompose, build_eq_table, eq_eval, products_except_self,
        unsafe_allocate_zero_vec,
    },
};
use crate::{
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    poly_iop::combined_check::util::{compute_leaves, extrapolate},
    util::{
        arithmetic::barycentric_weights,
        transcript::{TranscriptRead, TranscriptWrite},
    },
    Math, PolyIOP, PolyIOPErrors,
};
use ark_std::start_timer;
use ark_std::{end_timer, sync::Arc};
use deNetwork::DeMultiNet;
use ff::{BatchInverter, PrimeField};
use itertools::{izip, zip_eq, Itertools};
use serde::Serialize;
use serde::{de::DeserializeOwned, Deserialize};
use std::{iter, marker::PhantomData};
mod gaussian_elimination;
mod unipoly;
mod util;
use ark_std::mem::take;
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet, Stats};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
/// A `enum` specifying the possible failure modes of the arithmetics.
#[derive(Debug)]
pub enum ArithErrors {
    /// Invalid parameters: {0}
    InvalidParameters(String),
    /// Should not arrive to this point
    ShouldNotArrive,
    /// An error during (de)serialization: {0}
    SerializationErrors(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ZerocheckInstanceProof<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) polys: Vec<UniPoly<F>>,
    _marker: PhantomData<Pcs>,
}

impl<F: PrimeField, Pcs: PolynomialCommitmentScheme<F>> ZerocheckInstanceProof<F, Pcs> {
    pub fn new(polys: Vec<UniPoly<F>>) -> ZerocheckInstanceProof<F, Pcs> {
        ZerocheckInstanceProof {
            polys,
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck
    /// protocol: g_v(r_v) = oracle_g(r), as the oracle is not passed in.
    /// Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to
    ///   bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate
    ///   polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    /// #[tracing::instrument(skip_all, name = "Sumcheck::verify")]
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        zerocheck_r: &[F],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
    ) -> Result<(F, Vec<F>), PolyIOPErrors> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.polys.len(), num_rounds);
        for i in 0..self.polys.len() {
            let poly = &self.polys[i];
            // append the prover's message to the transcript
            // transcript.append_serializable_element(b"poly", poly)?;
            let _ = transcript.common_field_elements(&poly.coeffs);

            // verify degree bound
            if poly.degree() != degree_bound {
                return Err(PolyIOPErrors::InvalidProof(format!(
                    "degree_bound = {}, poly.degree() = {}",
                    degree_bound,
                    poly.degree(),
                )));
            }

            if poly.coeffs[0] + zerocheck_r[i] * (poly.coeffs.iter().skip(1).sum::<F>()) != e {
                return Err(PolyIOPErrors::InvalidProof(
                    "Inconsistent message".to_string(),
                ));
            }

            // derive the verifier's challenge for the next round
            // let r_i = transcript.get_and_append_challenge(b"challenge_nextround")?;

            let r_i = transcript.squeeze_challenge();
            // transcript.write_field_element(&r_i);

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}

pub trait CombinedCheck<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    type MultilinearExtension;
    type CombinedCheckSubClaim;
    // type CombinedCheckProof: Serialize + DeserializeOwned;
    type CombinedCheckProof;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a CombinedCheck
    /// is an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// CombinedCheck prover/verifier.

    fn prove_prepare(
        prover_param: &Pcs::ProverParam,
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
    ) -> Result<(MultilinearPolynomial<F>, Pcs::Commitment, F, F), PolyIOPErrors>;

    fn prove(
        to_prove: (MultilinearPolynomial<F>, Pcs::Commitment, F, F),
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        selectors: &[Self::MultilinearExtension],
        gate: &[(Option<usize>, Vec<usize>)],
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
    ) -> Result<(Self::CombinedCheckProof, MultilinearPolynomial<F>, Vec<F>), PolyIOPErrors>;

    fn verify(
        proof: &Self::CombinedCheckProof,
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
    ) -> Result<Self::CombinedCheckSubClaim, PolyIOPErrors>;

    fn check_openings(
        subclaim: &Self::CombinedCheckSubClaim,
        witness_openings: &[F],
        perm_openings: &[F],
        selector_openings: &[F],
        h_opening: &F,
        gate: &[(Option<usize>, Vec<usize>)],
    ) -> Result<(), PolyIOPErrors>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombinedCheckProof<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub proof: (ZerocheckInstanceProof<F, Pcs>, Vec<F>),
    pub h_comm: Pcs::Commitment,
    pub num_rounds: usize,
    pub degree_bound: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CombinedCheckSubClaim<F: PrimeField> {
    pub point: Vec<F>,
    pub zerocheck_expected_evaluation: F,
    pub h_expected_evaluation: F,
    pub zerocheck_r: Vec<F>,
    pub coeff: F,
    pub beta: F,
    pub gamma: F,
}

struct CombinedCheckInfo<'a, F: PrimeField> {
    num_witnesses: usize,
    num_selectors: usize,
    gate: &'a [(Option<usize>, Vec<usize>)],
    coeff: F,
    sid_offset: F,
}

// The first element in values is the eq_eval, second element is h
fn combined_check_combine_permcheck<F: PrimeField>(
    mut sid: F,
    values: &[F],
    info: &CombinedCheckInfo<F>,
) -> F {
    let witnesses = &values[..info.num_witnesses];
    let perms = &values[info.num_witnesses..2 * info.num_witnesses];
    let h_eval = values.last().unwrap();

    let mut g_evals = vec![F::ZERO; witnesses.len() * 2];
    for i in 0..witnesses.len() {
        // sid contains beta & gamma
        g_evals[i] = witnesses[i] + sid;
        sid += info.sid_offset;
    }
    for i in 0..witnesses.len() {
        // perm contains beta & gamma
        g_evals[witnesses.len() + i] = witnesses[i] + perms[i];
    }
    let g_products = products_except_self(&g_evals);

    // g_products is the product of all the g except self
    let mut sum = F::ZERO;
    for g_product in &g_products[..witnesses.len()] {
        sum += g_product;
    }
    for g_product in &g_products[witnesses.len()..] {
        sum -= g_product;
    }

    *h_eval * g_products[0] * g_evals[0] - sum
}

fn combined_check_combine_zerocheck<F: PrimeField>(
    _sid: F,
    values: &[F],
    info: &CombinedCheckInfo<F>,
) -> F {
    let witnesses = &values[..info.num_witnesses];
    let selectors = &values[2 * info.num_witnesses..2 * info.num_witnesses + info.num_selectors];

    info.gate
        .iter()
        .map(|(selector, witness_indices)| {
            let mut product = F::ONE;
            if let Some(selector_idx) = selector {
                product *= selectors[*selector_idx];
            }
            for witness_idx in witness_indices {
                product *= witnesses[*witness_idx];
            }
            product
        })
        .sum::<F>()
}

fn combined_sumcheck_prove_step<'a, F: PrimeField, Func1, Func2>(
    polys: &mut Vec<MultilinearPolynomial<F>>,
    eq_table: &[F],
    sid: F,
    multiplier: F,
    comb_func_1: &Func1,
    combined_degree_1: usize,
    comb_func_2: &Func2,
    combined_degree_2: usize,
    combine_coeff: F,
    mut previous_claim_1: &'a mut F,
    mut previous_claim_2: &'a mut F,
    r: F,
    r_inv: F,
    extrapolation_aux: &(Vec<F>, Vec<F>),
) -> (Vec<F>, Vec<F>, Vec<F>, F)
where
    Func1: Fn(F, &[F]) -> F + std::marker::Sync,
    Func2: Fn(F, &[F]) -> F + std::marker::Sync,
{
    let start = start_timer!(|| "zero check step");
    // Vector storing evaluations of combined polynomials g(x) = P_0(x) * ...
    // P_{num_polys} (x) for points {0, ..., |g(x)|}
    let mle_half = polys[0].evals.len() / 2;

    let (mut accum1, mut accum2, h_eval) = (0..mle_half)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![F::ZERO; polys.len()],
                    vec![F::ZERO; polys.len()],
                    vec![F::ZERO; combined_degree_1 + 1],
                    vec![F::ZERO; combined_degree_2 + 1],
                    F::ZERO,
                )
            },
            |(mut eval, mut step, mut acc1, mut acc2, mut h_eval), b| {
                let mut sid = multiplier * F::from(2 * (b as u64)) + sid;

                let eq_eval = eq_table[b];
                izip!(eval.iter_mut(), step.iter_mut(), polys.iter()).for_each(
                    |(eval, step, poly)| {
                        *eval = poly[b << 1];
                        *step = poly[(b << 1) + 1] - poly[b << 1];
                    },
                );
                acc1[0] += comb_func_1(sid, &eval) * eq_eval;
                acc2[0] += comb_func_2(sid, &eval) * eq_eval;
                h_eval += eval.last().unwrap();

                eval.iter_mut()
                    .zip(step.iter())
                    .for_each(|(eval, step)| *eval += step as &_);
                sid += multiplier;
                for eval_i in 2..(std::cmp::max(combined_degree_1, combined_degree_2) + 1) {
                    eval.iter_mut()
                        .zip(step.iter())
                        .for_each(|(eval, step)| *eval += step as &_);
                    sid += multiplier;

                    if eval_i < acc1.len() {
                        acc1[eval_i] += comb_func_1(sid, &eval) * eq_eval;
                    }
                    if eval_i < acc2.len() {
                        acc2[eval_i] += comb_func_2(sid, &eval) * eq_eval;
                    }
                }
                (eval, step, acc1, acc2, h_eval)
            },
        )
        .map(|(_, _, partial1, partial2, partial_heval)| (partial1, partial2, partial_heval))
        .reduce(
            || {
                (
                    vec![F::ZERO; combined_degree_1 + 1],
                    vec![F::ZERO; combined_degree_2 + 1],
                    F::ZERO,
                )
            },
            |(mut sum1, mut sum2, h_eval), (partial1, partial2, partial_h_eval)| {
                sum1.iter_mut()
                    .zip(partial1.iter())
                    .for_each(|(sum, partial)| *sum += partial);
                sum2.iter_mut()
                    .zip(partial2.iter())
                    .for_each(|(sum, partial)| *sum += partial);
                (sum1, sum2, h_eval + partial_h_eval)
            },
        );

    let mut should_swap = accum1.len() < accum2.len();
    if should_swap {
        (accum1, accum2) = (accum2, accum1);
        (previous_claim_1, previous_claim_2) = (previous_claim_2, previous_claim_1);
    }
    accum1[1] = r_inv * (*previous_claim_1 - (F::ONE - r) * accum1[0]);
    accum2[1] = r_inv * (*previous_claim_2 - (F::ONE - r) * accum2[0]);

    let (points, weights) = extrapolation_aux;
    let evals = accum1
        .iter()
        .zip(
            accum2
                .iter()
                .map(|x| *x)
                .chain((accum2.len()..accum1.len()).map(|i| {
                    let at = F::from(i as u64);
                    extrapolate(points, weights, &accum2, &at)
                })),
        )
        .map(|(sum1, sum2)| {
            if should_swap {
                combine_coeff * *sum1 + sum2
            } else {
                *sum1 + combine_coeff * sum2
            }
        })
        .collect::<Vec<_>>();

    end_timer!(start);
    if should_swap {
        (evals, accum2, accum1, h_eval)
    } else {
        (evals, accum1, accum2, h_eval)
    }
}

fn combined_sumcheck_prove<
    F: PrimeField + Serialize + DeserializeOwned,
    Func1,
    Func2,
    Pcs: PolynomialCommitmentScheme<F>,
>(
    claim_1: &F,
    claim_2: &F,
    num_rounds: usize,
    mut multiplier: F,
    mut sid: F,
    polys: &mut Vec<MultilinearPolynomial<F>>,
    zerocheck_r: &[F],
    comb_func_1: Func1,
    combined_degree_1: usize,
    comb_func_2: Func2,
    combined_degree_2: usize,
    extrapolation_aux: &(Vec<F>, Vec<F>),
    combine_coeff: F,
    transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
) -> ((ZerocheckInstanceProof<F, Pcs>, Vec<F>), Vec<F>, Vec<F>)
where
    Func1: Fn(F, &[F]) -> F + std::marker::Sync,
    Func2: Fn(F, &[F]) -> F + std::marker::Sync,
{
    let mut zerocheck_r_inv = zerocheck_r.to_vec();
    let mut scratch_space = vec![F::ZERO; zerocheck_r_inv.len()];
    BatchInverter::invert_with_external_scratch(&mut zerocheck_r_inv, &mut scratch_space);

    let eq_table = build_eq_table(&zerocheck_r, F::ONE);

    let mut r: Vec<F> = Vec::new();
    let mut proof_polys: Vec<UniPoly<F>> = Vec::new();
    let mut proof_h_evals = Vec::new();

    let mut previous_claim_1 = claim_1.clone();
    let mut previous_claim_2 = claim_2.clone();

    for round in 0..num_rounds {
        let (eval_points, evals_1, evals_2, eval_h) = combined_sumcheck_prove_step(
            polys,
            &eq_table[round],
            sid,
            multiplier,
            &comb_func_1,
            combined_degree_1,
            &comb_func_2,
            combined_degree_2,
            combine_coeff,
            &mut previous_claim_1,
            &mut previous_claim_2,
            zerocheck_r[round],
            zerocheck_r_inv[round],
            extrapolation_aux,
        );

        let step = start_timer!(|| "from evals");
        let round_uni_poly = UniPoly::from_evals(&eval_points);
        end_timer!(step);

        // append the prover's message to the transcript
        // transcript
        //     .append_serializable_element(b"poly", &round_uni_poly)
        //     .unwrap();
        transcript.write_field_elements(&round_uni_poly.coeffs);
        // transcript.write_field_element(&eval_h);
        // transcript.append_field_element(b"eval_h", &eval_h).unwrap();
        let r_j = transcript.squeeze_challenge();
        // transcript.write_field_element(&r_j);
        // let r_j = transcript
        //     .get_and_append_challenge(b"challenge_nextround")
        //     .unwrap();
        r.push(r_j);

        sid += r_j * multiplier;
        // multiplier.double_in_place();
        multiplier = multiplier * multiplier;

        // bound all tables to the verifier's challenege
        let step = start_timer!(|| "bind polys");
        let concurrency = (rayon::current_num_threads() * 2 + polys.len() - 1) / polys.len();
        polys
            .par_iter_mut()
            .for_each(|poly| bind_poly_var_bot_par(poly, &r_j, concurrency));
        rayon::join(
            || previous_claim_1 = interpolate_uni_poly(&evals_1, r_j),
            || previous_claim_2 = interpolate_uni_poly(&evals_2, r_j),
        );
        proof_polys.push(round_uni_poly);
        proof_h_evals.push(eval_h);
        end_timer!(step);
    }

    let final_evals = polys.iter().map(|poly| poly[0]).collect::<Vec<_>>();

    (
        (ZerocheckInstanceProof::new(proof_polys), proof_h_evals),
        r,
        final_evals,
    )
}

fn d_combined_sumcheck_prove<
    F: PrimeField + Serialize + DeserializeOwned,
    Func1,
    Func2,
    Pcs: PolynomialCommitmentScheme<F>,
>(
    claim_1: &F,
    claim_2: &F,
    num_rounds: usize,
    polys: &mut Vec<MultilinearPolynomial<F>>,
    zerocheck_r: &[F],
    mut multiplier: F,
    sid_init: F,
    comb_func_1: Func1,
    combined_degree_1: usize,
    comb_func_2: Func2,
    combined_degree_2: usize,
    extrapolation_aux: &(Vec<F>, Vec<F>),
    combine_coeff: F,
    transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
) -> Option<((ZerocheckInstanceProof<F, Pcs>, Vec<F>), Vec<F>, Vec<F>)>
where
    Func1: Fn(F, &[F]) -> F + std::marker::Sync,
    Func2: Fn(F, &[F]) -> F + std::marker::Sync,
{
    let num_party_vars = Net::n_parties().log_2();

    let index_vec: Vec<F> = bit_decompose(Net::party_id() as u64, num_party_vars)
        .into_iter()
        .map(|x| F::from(x as u64))
        .collect();
    let coeff = eq_eval(&zerocheck_r[num_rounds..], &index_vec).unwrap();

    let eq_table = build_eq_table(&zerocheck_r[..num_rounds], coeff);
    let mut zerocheck_r_inv = zerocheck_r[..num_rounds].to_vec();
    let mut scratch_space = vec![F::ZERO; zerocheck_r_inv.len()];
    BatchInverter::invert_with_external_scratch(&mut zerocheck_r_inv, &mut scratch_space);

    let mut r: Vec<F> = Vec::new();
    let mut proof_polys: Vec<UniPoly<F>> = Vec::new();
    let mut proof_h_evals = Vec::new();

    let mut previous_claim_1 = claim_1.clone();
    let mut previous_claim_2 = claim_2.clone();

    let mut sid = sid_init + multiplier * F::from((Net::party_id() * num_rounds.pow2()) as u64);

    for round in 0..num_rounds {
        let eval_points = combined_sumcheck_prove_step(
            polys,
            &eq_table[round],
            sid,
            multiplier,
            &comb_func_1,
            combined_degree_1,
            &comb_func_2,
            combined_degree_2,
            combine_coeff,
            &mut previous_claim_1,
            &mut previous_claim_2,
            zerocheck_r[round],
            zerocheck_r_inv[round],
            extrapolation_aux,
        );
        // let encoded = bincode::serialize(&eval_points)?;
        let all_eval_points = Net::send_to_master(&eval_points);

        // append the prover's message to the transcript
        let r_j = if Net::am_master() {
            let all_eval_points = all_eval_points.unwrap();
            let (eval_points, evals_1, evals_2, eval_h) = all_eval_points.iter().fold(
                (
                    vec![F::ZERO; all_eval_points[0].0.len()],
                    vec![F::ZERO; all_eval_points[0].1.len()],
                    vec![F::ZERO; all_eval_points[0].2.len()],
                    F::ZERO,
                ),
                |(mut evals, mut evals_1, mut evals_2, eval_h),
                 (partial, partial_1, partial_2, partial_h)| {
                    zip_eq(&mut evals, partial).for_each(|(acc, x)| *acc += *x);
                    zip_eq(&mut evals_1, partial_1).for_each(|(acc, x)| *acc += *x);
                    zip_eq(&mut evals_2, partial_2).for_each(|(acc, x)| *acc += *x);
                    (evals, evals_1, evals_2, eval_h + partial_h)
                },
            );

            let step = start_timer!(|| "from evals");
            let round_uni_poly = UniPoly::from_evals(&eval_points);
            end_timer!(step);

            // transcript
            //     .append_serializable_element(b"poly", &round_uni_poly)
            //     .unwrap();
            // transcript.append_field_element(b"eval_h", &eval_h).unwrap();

            // let r_j = transcript
            //     .get_and_append_challenge(b"challenge_nextround")
            //     .unwrap();
            transcript.write_field_elements(&round_uni_poly.coeffs);
            // transcript.write_field_element(&eval_h);
            let r_j = transcript.squeeze_challenge();
            // transcript.write_field_element(&r_j);
            rayon::join(
                || previous_claim_1 = interpolate_uni_poly(&evals_1, r_j),
                || previous_claim_2 = interpolate_uni_poly(&evals_2, r_j),
            );
            proof_polys.push(round_uni_poly);
            proof_h_evals.push(eval_h);
            Net::recv_from_master_uniform(Some(r_j))
        } else {
            Net::recv_from_master_uniform(None)
        };
        r.push(r_j);

        sid += r_j * multiplier;
        // multiplier.double_in_place();
        multiplier = multiplier * multiplier;

        // bound all tables to the verifier's challenege
        let step = start_timer!(|| "bind polys");
        let concurrency = (rayon::current_num_threads() * 2 + polys.len() - 1) / polys.len();
        polys
            .par_iter_mut()
            .for_each(|poly| bind_poly_var_bot_par(poly, &r_j, concurrency));
        end_timer!(step);
    }

    let final_evals = polys.iter().map(|poly| poly[0]).collect::<Vec<_>>();
    let all_final_evals = Net::send_to_master(&final_evals);

    if !Net::am_master() {
        return None;
    }

    let all_final_evals = all_final_evals.unwrap();
    let mut polys = (0..all_final_evals[0].len())
        .into_par_iter()
        .map(|poly_id| {
            MultilinearPolynomial::new(
                // num_party_vars,
                all_final_evals
                    .iter()
                    .map(|party_evals| party_evals[poly_id])
                    .collect(),
            )
        })
        .collect::<Vec<_>>();
    let ((mut proof, mut h_evals), mut r_final, final_evals) =
        combined_sumcheck_prove::<F, _, _, Pcs>(
            &previous_claim_1,
            &previous_claim_2,
            num_party_vars,
            multiplier,
            sid,
            &mut polys,
            &zerocheck_r[num_rounds..],
            comb_func_1,
            combined_degree_1,
            comb_func_2,
            combined_degree_2,
            extrapolation_aux,
            combine_coeff,
            transcript,
        );
    proof_polys.append(&mut proof.polys);
    proof_h_evals.append(&mut h_evals);
    r.append(&mut r_final);

    Some((
        (ZerocheckInstanceProof::new(proof_polys), proof_h_evals),
        r,
        final_evals,
    ))
}

fn combined_sumcheck_verify<F: PrimeField, Pcs: PolynomialCommitmentScheme<F>>(
    proof: &(ZerocheckInstanceProof<F, Pcs>, Vec<F>),
    num_rounds: usize,
    degree_bound: usize,
    zerocheck_r: &[F],
    transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
) -> Result<(F, F, Vec<F>), PolyIOPErrors> {
    let mut e = F::ZERO;
    let mut e2 = F::ZERO;
    let mut r: Vec<F> = Vec::new();

    let (proof, h_evals) = proof;
    // verify that there is a univariate polynomial for each round
    assert_eq!(proof.polys.len(), num_rounds);
    for i in 0..proof.polys.len() {
        let poly = &proof.polys[i];
        // append the prover's message to the transcript
        // transcript.append_serializable_element(b"poly", poly)?;
        let _ = transcript.common_field_elements(&poly.coeffs);
        // let _ = transcript.common_field_element(&h_evals[i]);
        // transcript.append_field_element(b"eval_h", &h_evals[i])?;

        // verify degree bound
        if poly.degree() != degree_bound {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "degree_bound = {}, poly.degree() = {}",
                degree_bound,
                poly.degree(),
            )));
        }

        if poly.coeffs[0] + zerocheck_r[i] * (poly.coeffs.iter().skip(1).sum::<F>()) != e {
            return Err(PolyIOPErrors::InvalidProof(
                "Inconsistent message".to_string(),
            ));
        }

        // derive the verifier's challenge for the next round
        // let r_i = transcript.get_and_append_challenge(b"challenge_nextround")?;
        let r_i = transcript.squeeze_challenge();

        r.push(r_i);

        // evaluate the claimed degree-ell polynomial at r_i
        e = poly.evaluate(&r_i);

        // (eval_h) + r_i (eval_1 - eval_h), eval_1 = e2 - eval_h
        e2 = h_evals[i] + r_i * (e2 - h_evals[i] - h_evals[i]);
    }

    Ok((e, e2, r))
}

impl<F, Pcs> CombinedCheck<F, Pcs> for PolyIOP<F>
where
    F: PrimeField + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    type MultilinearExtension = MultilinearPolynomial<F>;
    type CombinedCheckSubClaim = CombinedCheckSubClaim<F>;
    type CombinedCheckProof = CombinedCheckProof<F, Pcs>;

    fn prove_prepare(
        prover_param: &Pcs::ProverParam,
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
    ) -> Result<(MultilinearPolynomial<F>, Pcs::Commitment, F, F), PolyIOPErrors> {
        let start = start_timer!(|| "prove prepare");
        // let beta = transcript.get_and_append_challenge(b"beta")?;
        // let gamma = transcript.get_and_append_challenge(b"gamma")?;
        let beta = transcript.squeeze_challenge();
        // transcript.write_field_element(&beta);
        let gamma = transcript.squeeze_challenge();
        // transcript.write_field_element(&gamma);

        let mut leaves =
            compute_leaves::<F, false>(&beta, &gamma, witness, witness, perms).unwrap();
        let leaves_len = leaves.len();
        let mut leave = take(&mut leaves[0]);
        assert_eq!(leaves_len, 1);

        let half_len = leave.len() / 2;
        let nv = leave[0].len().log_2();
        leave.par_iter_mut().for_each(|evals| {
            // batch_inversion(evals);
            let mut scratch_space = vec![F::ZERO; evals.len()];
            BatchInverter::invert_with_external_scratch(evals, &mut scratch_space);
        });
        let h_evals = (0..leave[0].len())
            .into_par_iter()
            .map(|i| {
                leave[..half_len].iter().map(|eval| eval[i]).sum::<F>()
                    - leave[half_len..].iter().map(|eval| eval[i]).sum::<F>()
            })
            .collect::<Vec<_>>();
        let h_poly: MultilinearPolynomial<F> = MultilinearPolynomial::new(h_evals);
        let h_comm = Pcs::commit_and_write(prover_param, &h_poly, transcript).unwrap();

        end_timer!(start);

        Ok((h_poly, h_comm, beta, gamma))
    }

    fn prove(
        to_prove: (MultilinearPolynomial<F>, Pcs::Commitment, F, F),
        witness: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        selectors: &[Self::MultilinearExtension],
        gate: &[(Option<usize>, Vec<usize>)],
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
    ) -> Result<(Self::CombinedCheckProof, MultilinearPolynomial<F>, Vec<F>), PolyIOPErrors> {
        let start = start_timer!(|| "combined_check prove");

        let num_vars = to_prove.0.num_vars;
        let evals_len = witness[0].evals.len();

        // let r = transcript.get_and_append_challenge_vectors(b"0check r", num_vars)?;
        let r = transcript.squeeze_challenges(num_vars);
        // transcript.write_field_elements(&r);
        // let coeff = transcript.get_and_append_challenge(b"coeff")?;
        let coeff = transcript.squeeze_challenge();
        // transcript.write_field_element(&coeff);

        let num_witnesses = witness.len();
        let (h_poly, h_comm, beta, gamma) = to_prove;
        let (mut polys, h_poly_clone) = rayon::join(
            || {
                witness
                    .par_iter()
                    .map(|poly| MultilinearPolynomial::new(poly.evals.clone()))
                    .chain(perms.par_iter().map(|poly| {
                        MultilinearPolynomial::new(
                            // num_vars,
                            poly.iter().map(|x| *x * beta + gamma).collect(),
                        )
                    }))
                    .chain(
                        selectors
                            .par_iter()
                            .map(|poly| MultilinearPolynomial::new(poly.evals.clone())),
                    )
                    .collect::<Vec<_>>()
            },
            || MultilinearPolynomial::new(h_poly.evals.clone()),
        );
        polys.push(h_poly_clone);
        let max_gate_degree = gate
            .iter()
            .map(|(selector, witnesses)| {
                if *selector == None {
                    witnesses.len()
                } else {
                    witnesses.len() + 1
                }
            })
            .max()
            .unwrap();

        let info = CombinedCheckInfo {
            coeff,
            num_witnesses,
            num_selectors: selectors.len(),
            gate,
            sid_offset: beta * F::from(evals_len as u64),
        };

        let degree_zerocheck = max_gate_degree;
        let degree_permcheck = 2 * num_witnesses + 1;
        let extrapolation_aux = {
            let degree = std::cmp::min(degree_zerocheck, degree_permcheck);
            let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
            let weights = barycentric_weights(&points);
            (points, weights)
        };
        let (proof, point, _) = combined_sumcheck_prove::<F, _, _, Pcs>(
            &F::ZERO,
            &F::ZERO,
            num_vars,
            beta,
            gamma,
            &mut polys,
            &r,
            |sid, evals| combined_check_combine_zerocheck(sid, evals, &info),
            degree_zerocheck,
            |sid, evals| combined_check_combine_permcheck(sid, evals, &info),
            degree_permcheck,
            &extrapolation_aux,
            coeff,
            transcript,
        );

        end_timer!(start);

        Ok((
            CombinedCheckProof {
                proof,
                h_comm,
                num_rounds: num_vars,
                degree_bound: std::cmp::max(degree_zerocheck, degree_permcheck),
            },
            h_poly,
            point,
        ))
    }

    fn verify(
        proof: &Self::CombinedCheckProof,
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
    ) -> Result<Self::CombinedCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "combined_check verify");

        let beta = transcript.squeeze_challenge();
        let gamma = transcript.squeeze_challenge();
        // let zerocheck_r =
        //     transcript.get_and_append_challenge_vectors(b"0check r", proof.num_rounds)?;
        // let coeff = transcript.get_and_append_challenge(b"coeff")?;

        let zerocheck_r = transcript.squeeze_challenges(proof.num_rounds);
        let coeff = transcript.squeeze_challenge();

        let (zerocheck_expected_evaluation, h_expected_evaluation, point) =
            combined_sumcheck_verify(
                &proof.proof,
                proof.num_rounds,
                proof.degree_bound,
                &zerocheck_r,
                transcript,
            )?;

        end_timer!(start);

        Ok(CombinedCheckSubClaim {
            zerocheck_expected_evaluation,
            h_expected_evaluation,
            point,
            zerocheck_r,
            coeff,
            beta,
            gamma,
        })
    }

    fn check_openings(
        subclaim: &Self::CombinedCheckSubClaim,
        witness_openings: &[F],
        perm_openings: &[F],
        selector_openings: &[F],
        h_opening: &F,
        gate: &[(Option<usize>, Vec<usize>)],
    ) -> Result<(), PolyIOPErrors> {
        if *h_opening != subclaim.h_expected_evaluation {
            return Err(PolyIOPErrors::InvalidVerifier(
                "wrong subclaim on h".to_string(),
            ));
        }

        let nv = subclaim.point.len();
        let num_constraints = nv.pow2();
        let info = CombinedCheckInfo {
            gate,
            num_witnesses: witness_openings.len(),
            num_selectors: selector_openings.len(),
            coeff: subclaim.coeff,
            sid_offset: subclaim.beta * F::from(num_constraints as u64),
        };
        let mut evals = witness_openings
            .iter()
            .map(|x| *x)
            .chain(
                perm_openings
                    .iter()
                    .map(|perm| *perm * subclaim.beta + subclaim.gamma),
            )
            .chain(selector_openings.iter().map(|x| *x))
            .collect::<Vec<_>>();
        evals.push(*h_opening);
        let mut sid_eval = subclaim.gamma;
        let mut multiplier = subclaim.beta;
        for r in &subclaim.point {
            sid_eval += *r * multiplier;
            multiplier = multiplier * multiplier;
        }

        if combined_check_combine_zerocheck(sid_eval, &evals, &info)
            + info.coeff * combined_check_combine_permcheck(sid_eval, &evals, &info)
            != subclaim.zerocheck_expected_evaluation
        {
            return Err(PolyIOPErrors::InvalidVerifier(
                "wrong subclaim on zerocheck".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn evaluate_opt<F: PrimeField>(poly: &MultilinearPolynomial<F>, point: &[F]) -> F {
    assert_eq!(poly.num_vars, point.len());
    fix_variables(poly, point).evals[0]
}

pub fn fix_variables<F: PrimeField>(
    poly: &MultilinearPolynomial<F>,
    partial_point: &[F],
) -> MultilinearPolynomial<F> {
    assert!(
        partial_point.len() <= poly.num_vars,
        "invalid size of partial point"
    );
    let nv = poly.num_vars;
    let mut poly = poly.evals.to_vec();
    let dim = partial_point.len();
    // evaluate single variable of partial point from left to right
    for (i, point) in partial_point.iter().enumerate().take(dim) {
        poly = fix_one_variable_helper(&poly, nv - i, point);
    }

    MultilinearPolynomial::<F>::new((&poly[..(1 << (nv - dim))]).to_vec())
}

fn fix_one_variable_helper<F: PrimeField>(data: &[F], nv: usize, point: &F) -> Vec<F> {
    let mut res = unsafe_allocate_zero_vec::<F>(1 << (nv - 1));

    // evaluate single variable of partial point from left to right
    #[cfg(not(feature = "parallel"))]
    for i in 0..(1 << (nv - 1)) {
        res[i] = data[i] + (data[(i << 1) + 1] - data[i << 1]) * point;
    }

    #[cfg(feature = "parallel")]
    res.par_iter_mut().enumerate().for_each(|(i, x)| {
        *x = data[i << 1] + (data[(i << 1) + 1] - data[i << 1]) * point;
    });

    res
}

#[cfg(test)]
mod test {
    use super::{CombinedCheck, CombinedCheckProof};
    use crate::{
        pcs::{
            multilinear::{self, BasefoldExtParams},
            PolynomialCommitmentScheme,
        },
        poly::multilinear::MultilinearPolynomial,
        poly_iop::{combined_check::evaluate_opt, errors::PolyIOPErrors, PolyIOP},
        util::{
            goldilocksMont::GoldilocksMont,
            transcript::{
                Blake2sTranscript, FiatShamirTranscript, InMemoryTranscript, Keccak256Transcript,
                TranscriptWrite,
            },
        },
        Math,
    };
    use blake2::Blake2s256;
    use ff::Field;
    use ff::PrimeField;
    use rand::RngCore;
    use rand_chacha::rand_core::OsRng;
    use serde::{de::DeserializeOwned, Serialize};

    fn generate_polys<R: RngCore + Clone>(
        num_witnesses: usize,
        num_selectors: usize,
        nv: usize,
        gate: &[(Option<usize>, Vec<usize>)],
        rng: &mut R,
    ) -> (
        Vec<MultilinearPolynomial<GoldilocksMont>>,
        Vec<MultilinearPolynomial<GoldilocksMont>>,
        Vec<MultilinearPolynomial<GoldilocksMont>>,
    ) {
        let num_constraints = nv.pow2();
        let mut selectors: Vec<Vec<GoldilocksMont>> = vec![vec![]; num_selectors];
        let mut witnesses: Vec<Vec<GoldilocksMont>> = vec![vec![]; num_witnesses];

        for cs in 0..num_constraints {
            let mut cur_selectors: Vec<GoldilocksMont> = (0..(num_selectors - 1))
                .map(|_| GoldilocksMont::random(rng.clone()))
                .collect();
            let cur_witness: Vec<GoldilocksMont> = if cs < num_constraints / 4 {
                (0..num_witnesses)
                    .map(|_| GoldilocksMont::random(rng.clone()))
                    .collect()
            } else {
                let row = cs % (num_constraints / 4);
                (0..num_witnesses).map(|i| witnesses[i][row]).collect()
            };
            let mut last_selector = GoldilocksMont::ZERO;
            for (index, (q, wit)) in gate.iter().enumerate() {
                if index != num_selectors - 1 {
                    let mut cur_monomial = GoldilocksMont::ONE;
                    cur_monomial = match q {
                        Some(p) => cur_monomial * cur_selectors[*p],
                        None => cur_monomial,
                    };
                    for wit_index in wit.iter() {
                        cur_monomial *= cur_witness[*wit_index];
                    }
                    last_selector += cur_monomial;
                } else {
                    let mut cur_monomial = GoldilocksMont::ONE;
                    for wit_index in wit.iter() {
                        cur_monomial *= cur_witness[*wit_index];
                    }
                    let inv = (-cur_monomial).invert().unwrap();
                    last_selector *= inv;
                }
            }
            cur_selectors.push(last_selector);
            for i in 0..num_selectors {
                selectors[i].push(cur_selectors[i]);
            }
            for i in 0..num_witnesses {
                witnesses[i].push(cur_witness[i]);
            }
        }

        let permutation =
            (0..num_witnesses)
                .map(|witness_idx| {
                    let portion_len = num_constraints / 4;
                    (0..portion_len)
                        .map(|i| {
                            GoldilocksMont::from(
                                (witness_idx * num_constraints + i + portion_len) as u64,
                            )
                        })
                        .chain((0..portion_len).map(|i| {
                            GoldilocksMont::from(
                                (witness_idx * num_constraints + i + 3 * portion_len) as u64,
                            )
                        }))
                        .chain((0..portion_len).map(|i| {
                            GoldilocksMont::from((witness_idx * num_constraints + i) as u64)
                        }))
                        .chain((0..portion_len).map(|i| {
                            GoldilocksMont::from(
                                (witness_idx * num_constraints + i + 2 * portion_len) as u64,
                            )
                        }))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

        (
            witnesses
                .into_iter()
                .map(|vec| MultilinearPolynomial::new(vec))
                .collect(),
            permutation
                .into_iter()
                .map(|vec| MultilinearPolynomial::new(vec))
                .collect(),
            selectors
                .into_iter()
                .map(|vec| MultilinearPolynomial::new(vec))
                .collect(),
        )
    }

    fn test_combined_check_helper<
        F: PrimeField + Serialize + DeserializeOwned,
        PCS: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    >(
        witnesses: &[MultilinearPolynomial<F>],
        perms: &[MultilinearPolynomial<F>],
        selectors: &[MultilinearPolynomial<F>],
        gate: &[(Option<usize>, Vec<usize>)],
        pcs_param: &PCS::ProverParam,
        transcript: &mut impl TranscriptWrite<PCS::CommitmentChunk, F>,
    ) -> Result<(CombinedCheckProof<F, PCS>, MultilinearPolynomial<F>), PolyIOPErrors> {
        // prover

        // let mut transcript = Blake2sTranscript::new(());

        let to_prove = <PolyIOP<F> as CombinedCheck<F, PCS>>::prove_prepare(
            pcs_param, &witnesses, perms, transcript,
        )?;

        let (proof, h_poly, _) = <PolyIOP<F> as CombinedCheck<F, PCS>>::prove(
            to_prove, witnesses, perms, selectors, gate, transcript,
        )?;

        // // verifier
        // let mut re_transcript = Blake2sTranscript::from_proof((), transcript.into_proof());
        // let subclaim =
        //     <PolyIOP<F> as CombinedCheck<F, PCS>>::verify(&proof, &mut re_transcript)?;

        // let witness_openings = witnesses
        //     .iter()
        //     .map(|f| evaluate_opt(f, &subclaim.point))
        //     .collect::<Vec<_>>();
        // let perm_openings = perms
        //     .iter()
        //     .map(|f| evaluate_opt(f, &subclaim.point))
        //     .collect::<Vec<_>>();
        // let selector_openings = selectors
        //     .iter()
        //     .map(|f| evaluate_opt(f, &subclaim.point))
        //     .collect::<Vec<_>>();
        // let h_opening = evaluate_opt(&h_poly, &subclaim.point);

        // <PolyIOP<E::ScalarField> as CombinedCheck<E, PCS>>::check_openings(
        //     &subclaim,
        //     &witness_openings,
        //     &perm_openings,
        //     &selector_openings,
        //     &h_opening,
        //     gate,
        // )
        Ok((proof, h_poly))
    }

    fn test_combined_check(nv: usize) -> Result<(), PolyIOPErrors> {
        // let mut rng = test_rng();

        // let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, nv)?;
        // let (pcs_param, _) = MultilinearKzgPCS::<Bn254>::trim(&srs, None, Some(nv))?;
        type base = multilinear::Basefold<GoldilocksMont, Blake2s256, Nineteen8>;
        let num_vars = nv;
        let poly_size = 1 << num_vars;

        // let gate = vec![
        //     (Some(0), vec![0]),
        //     (Some(1), vec![1]),
        //     (Some(2), vec![2]),
        //     (Some(3), vec![3]),
        //     (Some(4), vec![0, 1]),
        //     (Some(5), vec![2, 3]),
        //     (Some(6), vec![0, 0, 0, 0, 0]),
        //     (Some(7), vec![1, 1, 1, 1, 1]),
        //     (Some(8), vec![2, 2, 2, 2, 2]),
        //     (Some(9), vec![3, 3, 3, 3, 3]),
        //     (Some(10), vec![0, 1, 2, 3]),
        //     (Some(11), vec![4]),
        //     (Some(12), vec![]),
        // ];
        let gate = vec![
            (Some(0), vec![0]),
            (Some(1), vec![1]),
            (Some(2), vec![0, 1]),
            (Some(3), vec![2]),
            (Some(4), vec![]),
        ];
        let mut rng = OsRng;
        let (witnesses, perms, selectors) = generate_polys(5, 13, num_vars, &gate, &mut rng);

        let (pp, vp) = {
            let poly_size = 1 << num_vars;
            let param = base::setup(poly_size, 1, &mut rng).unwrap();
            println!("before trim");
            base::trim(&param, poly_size, 1).unwrap()
        };
        let mut transcript: FiatShamirTranscript<_, _> = Blake2sTranscript::new(());
        let (proof, h_poly) = test_combined_check_helper::<GoldilocksMont, base>(
            &witnesses,
            &perms,
            &selectors,
            &gate,
            &pp,
            &mut transcript,
        )
        .unwrap();

        // // verifier
        let mut transcript = Blake2sTranscript::from_proof((), &transcript.into_proof());
        let subclaim = <PolyIOP<GoldilocksMont> as CombinedCheck<GoldilocksMont, base>>::verify(
            &proof,
            &mut transcript,
        )?;

        let witness_openings = witnesses
            .iter()
            .map(|f| evaluate_opt(f, &subclaim.point))
            .collect::<Vec<_>>();
        let perm_openings = perms
            .iter()
            .map(|f| evaluate_opt(f, &subclaim.point))
            .collect::<Vec<_>>();
        let selector_openings = selectors
            .iter()
            .map(|f| evaluate_opt(f, &subclaim.point))
            .collect::<Vec<_>>();
        let h_opening = evaluate_opt(&h_poly, &subclaim.point);

        let _ = <PolyIOP<GoldilocksMont> as CombinedCheck<GoldilocksMont, base>>::check_openings(
            &subclaim,
            &witness_openings,
            &perm_openings,
            &selector_openings,
            &h_opening,
            &gate,
        );

        Ok(())
    }

    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        test_combined_check(5)
    }
    #[derive(Debug)]
    pub struct Nineteen8 {}
    impl BasefoldExtParams for Nineteen8 {
        fn get_rate() -> usize {
            return 3;
        }

        fn get_basecode_rounds() -> usize {
            return 1;
        }

        fn get_reps() -> usize {
            return 403;
        }
        fn get_rs_basecode() -> bool {
            true
        }
        fn get_code_type() -> String {
            "random".to_string().to_string()
        }
    }
}
