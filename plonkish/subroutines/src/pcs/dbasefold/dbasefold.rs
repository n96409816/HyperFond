use crate::{util::transcript::{
    Blake2s256Transcript, FiatShamirTranscript, FieldTranscript, FieldTranscriptRead,
    FieldTranscriptWrite, InMemoryTranscript, TranscriptRead, TranscriptWrite,
}, pcs::errors::PCSError};
use ark_bls12_381::Fr;
use ark_ff::{batch_inversion, BigInteger, PrimeField};
use ark_poly::{evaluations, DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use ark_std::start_timer;
use blake2::{Blake2s256, Digest};
use crypto_common::{generic_array::GenericArray, KeyIvInit, Output, OutputSizeUser};
use ctr::cipher::{StreamCipher, StreamCipherSeek};
use ff::BatchInverter;
use itertools::Itertools;
use num_traits::Unsigned;
use rand_chacha::ChaCha8Rng;
use rand_core::{OsRng, RngCore, SeedableRng};
use rayon::{
    prelude::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};
use std::{
    marker::PhantomData,
    ptr::{swap, swap_nonoverlapping},
    slice,
    sync::Arc,
    time::Instant,
};

use crate::{
    pcs::{BasefoldExtParams, NewPolynomialCommitmentScheme},
    util::log2_strict,
};

#[derive(Default, Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BasefoldParams<F: PrimeField> {
    log_rate: usize,
    num_verifier_queries: usize,
    pub num_vars: usize,
    pub num_rounds: usize,
    table_w_weights: Vec<Vec<(F, F)>>,
    table: Vec<Vec<F>>,
    rs_basecode: bool,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BasefoldProverParams<F: PrimeField> {
    pub log_rate: usize,
    table_w_weights: Vec<Vec<(F, F)>>,
    pub table: Vec<Vec<F>>,
    num_verifier_queries: usize,
    pub num_vars: usize,
    num_rounds: usize,
    rs_basecode: bool,
    code_type: String,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BasefoldVerifierParams<F: PrimeField> {
    pub num_vars: usize,
    log_rate: usize,
    num_verifier_queries: usize,
    pub num_rounds: usize,
    table_w_weights: Vec<Vec<(F, F)>>,
    rs_basecode: bool,
    code_type: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Type1Polynomial<F: PrimeField> {
    pub poly: Vec<F>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct Type2Polynomial<F: PrimeField> {
    pub poly: Vec<F>,
}

// impl<F: PrimeField, H: Hash> CanonicalDeserialize for BasefoldCommitment<F,
// H> where
//     H: Hash,
//     Output<H>: Valid,
// {
//     // Implement deserialization logic
// }

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct BasefoldCommitment<F: PrimeField> {
    pub codeword: Type1Polynomial<F>,
    pub codeword_tree: Vec<Vec<[u8; 32]>>,
    pub bh_evals: Type1Polynomial<F>,
}

impl<F: PrimeField> Default for BasefoldCommitment<F> {
    fn default() -> Self {
        Self {
            codeword: Type1Polynomial { poly: Vec::new() },
            codeword_tree: vec![vec![[0; 32]]],
            bh_evals: Type1Polynomial { poly: Vec::new() },
        }
    }
}

impl<F: PrimeField> BasefoldCommitment<F> {
    fn from_root(root: [u8; 32]) -> Self {
        Self {
            codeword: Type1Polynomial { poly: Vec::new() },
            codeword_tree: vec![vec![root]],
            bh_evals: Type1Polynomial { poly: Vec::new() },
        }
    }
}
impl<F: PrimeField> PartialEq for BasefoldCommitment<F> {
    fn eq(&self, other: &Self) -> bool {
        self.codeword.poly.eq(&other.codeword.poly)
            && self.codeword_tree.eq(&other.codeword_tree)
            && self.bh_evals.poly.eq(&other.bh_evals.poly)
    }
}

impl<F: PrimeField> AsRef<[[u8; 32]]> for BasefoldCommitment<F> {
    fn as_ref(&self) -> &[[u8; 32]] {
        let root = &self.codeword_tree[self.codeword_tree.len() - 1][0];
        slice::from_ref(&root)
    }
}

impl<F: PrimeField> AsRef<[u8; 32]> for BasefoldCommitment<F> {
    fn as_ref(&self) -> &[u8; 32] {
        let root = &self.codeword_tree[self.codeword_tree.len() - 1][0];
        &root
    }
}

#[derive(Debug)]
pub struct Basefold<F: PrimeField, V: BasefoldExtParams>(PhantomData<(F, V)>);

impl<F: PrimeField, V: BasefoldExtParams> Clone for Basefold<F, V> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<F, V> NewPolynomialCommitmentScheme<F> for Basefold<F, V>
where
    F: PrimeField + CanonicalSerialize + CanonicalDeserialize,
    V: BasefoldExtParams,
{
    type Param = BasefoldParams<F>;
    type ProverParam = BasefoldProverParams<F>;
    type VerifierParam = BasefoldVerifierParams<F>;
    type Polynomial = Arc<DenseMultilinearExtension<F>>;
    type Point = Vec<F>;
    type Commitment = BasefoldCommitment<F>;
    type CommitmentChunk = [u8; 32];

    fn setup(
        poly_size: usize,
        batch_size: usize,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self::Param, PCSError> {
        let log_rate = V::get_rate();
        // let mut test_rng = ChaCha8Rng::from_entropy();
        let (table_w_weights, table) = get_table_aes(poly_size, log_rate, rng);
        let mut rs_basecode = false;
        if V::get_rs_basecode() == true && V::get_basecode_rounds() > 0 {
            rs_basecode = true;
        }
        Ok(BasefoldParams {
            log_rate,
            num_verifier_queries: V::get_reps(),
            num_vars: log2_strict(poly_size),
            num_rounds: log2_strict(poly_size) - V::get_basecode_rounds(),
            table_w_weights,
            table,
            rs_basecode,
        })
    }

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError> {
        let mut rounds = param.num_vars;
        // if param.num_rounds.is_some() {
        //     rounds = param.num_rounds.unwrap();
        // }
        rounds = param.num_rounds;

        Ok((
            BasefoldProverParams {
                log_rate: param.log_rate,
                table_w_weights: param.table_w_weights.clone(),
                table: param.table.clone(),
                num_verifier_queries: param.num_verifier_queries,
                num_vars: param.num_vars,
                num_rounds: rounds,
                rs_basecode: param.rs_basecode,
                code_type: V::get_code_type(),
            },
            BasefoldVerifierParams {
                num_vars: param.num_vars,
                log_rate: param.log_rate,
                num_verifier_queries: param.num_verifier_queries,
                num_rounds: rounds,
                table_w_weights: param.table_w_weights.clone(),
                rs_basecode: param.rs_basecode,
                code_type: V::get_code_type(),
            },
        ))
    }

    fn commit(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError> {
        let p = Type2Polynomial {
            poly: poly.evaluations.clone(),
        };
        let (coeffs, mut bh_evals) = interpolate_over_boolean_hypercube_with_copy(&p);

        let mut commitment = Type1Polynomial::default();
        commitment =
            evaluate_over_foldable_domain(pp.log_rate, coeffs, &pp.table, pp.code_type.clone());

        let tree = merkelize::<F>(&commitment);

        Ok(Self::Commitment {
            codeword: commitment,
            codeword_tree: tree,
            bh_evals,
        })
    }

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Self::Point,
        eval: &F,
        // transcript: &mut transcript::IOPTranscript<F>,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), PCSError> {
        let (trees, sum_check_oracles, mut oracles, mut bh_evals, mut eq, eval) = commit_phase(
            &point,
            &comm,
            transcript,
            pp.num_vars,
            pp.num_rounds,
            &pp.table_w_weights,
            pp.code_type.clone(),
            pp.log_rate,
        );

        let (queried_els, queries_usize_) =
            query_phase(transcript, &comm, &oracles, pp.num_verifier_queries);

        // a proof consists of roots, merkle paths, query paths, sum check oracles,
        // eval, and final oracle

        transcript.write_field_element(&eval); // write eval

        if pp.num_rounds < pp.num_vars {
            transcript.write_field_elements(&bh_evals.poly); // write bh_evals
            transcript.write_field_elements(&eq.poly); // write eq
        }

        // write final oracle
        let mut final_oracle = oracles.pop().unwrap();
        transcript.write_field_elements(&final_oracle.poly);

        // write query paths
        queried_els
            .iter()
            .map(|q| &q.0)
            .flatten()
            .for_each(|query| {
                transcript.write_field_element(&query.0);
                transcript.write_field_element(&query.1);
            });

        // write merkle paths
        queried_els.iter().for_each(|query| {
            let indices = &query.1;
            indices.into_iter().enumerate().for_each(|(i, q)| {
                if (i == 0) {
                    write_merkle_path::<F>(&comm.codeword_tree, *q, transcript);
                } else {
                    write_merkle_path::<F>(&trees[i - 1], *q, transcript);
                }
            })
        });

        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        // comm: &Self::Commitment,
        point: &Self::Point,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
        rng: &mut ChaCha8Rng,
    ) -> Result<(), PCSError> {
        let field_size = 255;
        let n = (1 << (vp.num_vars + vp.log_rate));
        // read first $(num_var - 1) commitments

        let mut fold_challenges: Vec<F> = Vec::with_capacity(vp.num_vars);
        let mut size = 0;
        let mut roots = Vec::new();
        let mut sum_check_oracles = Vec::new();
        for i in 0..vp.num_rounds {
            roots.push(transcript.read_commitment().unwrap());
            sum_check_oracles.push(transcript.read_field_elements(3).unwrap());
            fold_challenges.push(transcript.squeeze_challenge());
        }
        sum_check_oracles.push(transcript.read_field_elements(3).unwrap());

        let mut query_challenges = transcript.squeeze_challenges(vp.num_verifier_queries);

        size = size + field_size * (3 * (vp.num_rounds + 1));
        // read eval

        let eval = &transcript.read_field_element().unwrap(); // do not need eval in proof

        let mut bh_evals = Vec::new();
        let mut eq = Vec::new();
        if vp.num_rounds < vp.num_vars {
            bh_evals = transcript
                .read_field_elements(1 << (vp.num_vars - vp.num_rounds))
                .unwrap();
            eq = transcript
                .read_field_elements(1 << (vp.num_vars - vp.num_rounds))
                .unwrap();
            size = size + field_size * (bh_evals.len() + eq.len());
        }

        // read final oracle
        let mut final_oracle = transcript
            .read_field_elements(1 << (vp.num_vars - vp.num_rounds + vp.log_rate))
            .unwrap();

        size = size + field_size * final_oracle.len();
        // read query paths
        let num_queries = vp.num_verifier_queries * 2 * (vp.num_rounds + 1);

        let all_qs = transcript.read_field_elements(num_queries).unwrap();

        size = size + (num_queries - 2) * field_size;
        //        println!("size for all iop queries {:?}", size);

        let i_qs = all_qs.chunks((vp.num_rounds + 1) * 2).collect_vec();

        assert_eq!(i_qs.len(), vp.num_verifier_queries);

        let mut queries = i_qs.iter().map(|q| q.chunks(2).collect_vec()).collect_vec();

        assert_eq!(queries.len(), vp.num_verifier_queries);

        // read merkle paths

        let mut query_merkle_paths: Vec<Vec<Vec<Vec<[u8; 32]>>>> =
            Vec::with_capacity(vp.num_verifier_queries);
        let query_merkle_paths: Vec<Vec<Vec<Vec<[u8; 32]>>>> = (0..vp.num_verifier_queries)
            .into_iter()
            .map(|i| {
                let mut merkle_paths: Vec<Vec<Vec<[u8; 32]>>> =
                    Vec::with_capacity(vp.num_rounds + 1);
                for round in 0..(vp.num_rounds + 1) {
                    let mut merkle_path: Vec<[u8; 32]> = transcript
                        .read_commitments(2 * (vp.num_vars - round + vp.log_rate - 1))
                        .unwrap();
                    size = size + 256 * (2 * (vp.num_vars - round + vp.log_rate - 1));

                    let chunked_path: Vec<Vec<[u8; 32]>> =
                        merkle_path.chunks(2).map(|c| c.to_vec()).collect_vec();

                    merkle_paths.push(chunked_path);
                }
                merkle_paths
            })
            .collect();

        verifier_query_phase::<F>(
            &query_challenges,
            &query_merkle_paths,
            &sum_check_oracles,
            &fold_challenges,
            &queries,
            vp.num_rounds,
            vp.num_vars,
            vp.log_rate,
            &roots,
            rng.clone(),
            &eval,
            &vp.code_type,
        );
        let mut next_oracle = Type1Polynomial { poly: final_oracle };
        let mut bh_evals = Type1Polynomial { poly: bh_evals };
        let mut eq = Type1Polynomial { poly: eq };
        // if (!vp.rs_basecode) {
        virtual_open(
            vp.num_vars,
            vp.num_rounds,
            &mut eq,
            &mut bh_evals,
            &mut next_oracle,
            point,
            &mut fold_challenges,
            &vp.table_w_weights,
            &mut sum_check_oracles,
            vp.log_rate,
            vp.code_type.clone(),
        );
        // }
        Ok(())
    }
}

fn virtual_open<F: PrimeField>(
    num_vars: usize,
    num_rounds: usize,
    eq: &mut Type1Polynomial<F>,
    bh_evals: &mut Type1Polynomial<F>,
    last_oracle: &Type1Polynomial<F>,
    point: &Vec<F>,
    challenges: &mut Vec<F>,
    table: &Vec<Vec<(F, F)>>,
    sum_check_oracles: &mut Vec<Vec<F>>,
    log_rate: usize,
    code_type: String,
) {
    let mut rng = ChaCha8Rng::from_entropy();
    let rounds = num_vars - num_rounds;

    let mut oracles = Vec::with_capacity(rounds);
    let mut new_oracle = last_oracle;
    for round in 0..rounds {
        let challenge: F = rand_chacha(&mut rng);
        challenges.push(challenge);

        sum_check_oracles.push(sum_check_challenge_round(eq, bh_evals, challenge));

        oracles.push(basefold_one_round_by_interpolation_weights::<F>(
            &table,
            round + num_rounds,
            &new_oracle,
            challenge,
            num_vars,
            log_rate,
            code_type.clone(),
        ));
        new_oracle = &oracles[round];
    }

    let mut no = new_oracle.clone();
    no.poly.dedup();

    // verify it information-theoretically
    let mut eq_r_ = F::ONE;
    for i in 0..challenges.len() {
        eq_r_ = eq_r_ * (challenges[i] * point[i] + (F::ONE - challenges[i]) * (F::ONE - point[i]));
    }
    let last_challenge = challenges[challenges.len() - 1];

    // assert_eq!(
    //     degree_2_eval(&sum_check_oracles[challenges.len() - 1], last_challenge),
    //     eq_r_ * no.poly[0]
    // );
}

fn rand_chacha<F: PrimeField>(mut rng: &mut ChaCha8Rng) -> F {
    let bytes = (F::MODULUS_BIT_SIZE as usize).next_power_of_two() / 8;
    let mut dest: Vec<u8> = vec![0u8; bytes];
    rng.fill_bytes(&mut dest);
    from_raw_bytes::<F>(&dest)
}

fn verifier_query_phase<F: PrimeField>(
    query_challenges: &Vec<F>,
    query_merkle_paths: &Vec<Vec<Vec<Vec<[u8; 32]>>>>,
    sum_check_oracles: &Vec<Vec<F>>,
    fold_challenges: &Vec<F>,
    queries: &Vec<Vec<&[F]>>,
    num_rounds: usize,
    num_vars: usize,
    log_rate: usize,
    roots: &Vec<[u8; 32]>,
    rng: ChaCha8Rng,
    eval: &F,
    code_type: &str,
) -> Vec<usize> {
    println!("VERIFIER QUERY PHASE");
    println!("CHALLEGNES LENGTH {:?}", query_challenges.len());
    // assert_eq!(query_challenges.len(), 1);
    let n = (1 << (num_vars + log_rate));
    let mut queries_usize: Vec<usize> = query_challenges
        .par_iter()
        .map(|x_index| {
            let x_repr = (*x_index).into_bigint().to_bytes_be();
            let mut x: &[u8] = x_repr.as_ref();
            let (int_bytes, rest) = x.split_at(std::mem::size_of::<u32>());
            let x_int: u32 = u32::from_be_bytes(int_bytes.try_into().unwrap());
            ((x_int as usize) % n).into()
        })
        .collect();
    let mut key: [u8; 16] = [0u8; 16];
    let mut iv: [u8; 16] = [0u8; 16];
    let mut rng = rng.clone();
    rng.set_word_pos(0);
    rng.fill_bytes(&mut key);
    rng.fill_bytes(&mut iv);

    type Aes128Ctr64LE = ctr::Ctr32LE<aes::Aes128>;
    let mut cipher = Aes128Ctr64LE::new(
        GenericArray::from_slice(&key[..]),
        GenericArray::from_slice(&iv[..]),
    );
    queries_usize
        .iter_mut()
        .enumerate()
        .for_each(|(qi, query_index)| {
            let mut cipher = cipher.clone();
            let mut rng = rng.clone();
            let mut cur_index = *query_index;
            let mut cur_queries = &queries[qi];

            for i in 0..num_rounds {
                let temp = cur_index;
                let mut other_index = cur_index ^ 1;
                if (other_index < cur_index) {
                    cur_index = other_index;
                    other_index = temp;
                }

                assert_eq!(cur_index % 2, 0);

                let ri0 = reverse_bits(cur_index, num_vars + log_rate - i);

                let now = Instant::now();
                let mut x0 = F::ZERO;
                let mut x1 = F::ZERO;

                x0 = query_point(
                    1 << (num_vars + log_rate - i),
                    ri0,
                    &mut rng,
                    num_vars + log_rate - i - 1,
                    &mut cipher,
                );
                x1 = -x0;

                let res = interpolate2(
                    [(x0, cur_queries[i][0]), (x1, cur_queries[i][1])],
                    fold_challenges[i],
                );

                // assert_eq!(res, cur_queries[i + 1][(cur_index >> 1) % 2]);

                authenticate_merkle_path_root::<F>(
                    &query_merkle_paths[qi][i],
                    (cur_queries[i][0], cur_queries[i][1]),
                    cur_index,
                    &roots[i],
                );

                cur_index >>= 1;
            }
        });

    assert_eq!(eval, &degree_2_zero_plus_one(&sum_check_oracles[0]));

    for i in 0..fold_challenges.len() - 1 {
        let left = degree_2_eval(&sum_check_oracles[i], fold_challenges[i]);
        let right = degree_2_zero_plus_one(&sum_check_oracles[i + 1]);
        // assert_eq!(
        //     left,
        //     right
        // );
    }
    return queries_usize;
}

fn degree_2_eval<F: PrimeField>(poly: &Vec<F>, point: F) -> F {
    poly[0] + point * poly[1] + point * point * poly[2]
}

fn degree_2_zero_plus_one<F: PrimeField>(poly: &Vec<F>) -> F {
    poly[0] + poly[0] + poly[1] + poly[2]
}

fn authenticate_merkle_path_root<F: PrimeField>(
    path: &Vec<Vec<[u8; 32]>>,
    leaves: (F, F),
    mut x_index: usize,
    root: &[u8; 32],
) {
    let mut hasher = Blake2s256::default();
    let mut hash = [0; 32];
    hasher.update(&leaves.0.into_bigint().to_bytes_le());
    hasher.update(&leaves.1.into_bigint().to_bytes_le());
    hasher.finalize_into_reset((&mut hash).into());

    // assert_eq!(hash, path[0][(x_index >> 1) % 2]);
    x_index >>= 1;
    for i in 0..path.len() - 1 {
        let mut hasher = Blake2s256::default();
        let mut hash = [0; 32];
        hasher.update(&path[i][0]);
        hasher.update(&path[i][1]);
        hasher.finalize_into_reset((&mut hash).into());

        // assert_eq!(hash, path[i + 1][(x_index >> 1) % 2]);
        x_index >>= 1;
    }
    let mut hasher = Blake2s256::default();
    let mut hash = [0; 32];
    hasher.update(&path[path.len() - 1][0]);
    hasher.update(&path[path.len() - 1][1]);
    hasher.finalize_into_reset((&mut hash).into());
    // assert_eq!(&hash, root);
}

pub fn interpolate2<F: PrimeField>(points: [(F, F); 2], x: F) -> F {
    // a0 -> a1
    // b0 -> b1
    // x  -> a1 + (x-a0)*(b1-a1)/(b0-a0)
    let (a0, a1) = points[0];
    let (b0, b1) = points[1];
    assert_ne!(a0, b0);
    a1 + (x - a0) * (b1 - a1) * (b0 - a0).inverse().unwrap()
}

pub fn query_point<F: PrimeField>(
    block_length: usize,
    eval_index: usize,
    mut rng: &mut ChaCha8Rng,
    level: usize,
    mut cipher: &mut ctr::Ctr32LE<aes::Aes128>,
) -> F {
    let level_index = eval_index % (block_length);
    let mut el = query_root_table_from_rng_aes::<F>(
        level,
        (level_index % (block_length >> 1)),
        &mut rng,
        &mut cipher,
    );

    if level_index >= (block_length >> 1) {
        el = -F::ONE * el;
    }

    return el;
}

pub fn query_root_table_from_rng_aes<F: PrimeField>(
    level: usize,
    index: usize,
    rng: &mut ChaCha8Rng,
    cipher: &mut ctr::Ctr32LE<aes::Aes128>,
) -> F {
    let mut level_offset: u128 = 1;
    for lg_m in 1..=level {
        let half_m = 1 << (lg_m - 1);
        level_offset += half_m;
    }

    let pos = ((level_offset + (index as u128))
        * ((F::MODULUS_BIT_SIZE as usize).next_power_of_two() as u128))
        .checked_div(8)
        .unwrap();

    cipher.try_seek(pos);

    let bytes = (F::MODULUS_BIT_SIZE as usize).next_power_of_two() / 8;
    let mut dest: Vec<u8> = vec![0u8; bytes];
    cipher.apply_keystream(&mut dest);

    let res = from_raw_bytes::<F>(&dest);

    res
}

pub fn reverse_bits(n: usize, num_bits: usize) -> usize {
    // NB: The only reason we need overflowing_shr() here as opposed
    // to plain '>>' is to accommodate the case n == num_bits == 0,
    // which would become `0 >> 64`. Rust thinks that any shift of 64
    // bits causes overflow, even when the argument is zero.
    n.reverse_bits()
        .overflowing_shr(usize::BITS - num_bits as u32)
        .0
}

fn write_merkle_path<F: PrimeField>(
    tree: &Vec<Vec<[u8; 32]>>,
    mut x_index: usize,
    transcript: &mut impl TranscriptWrite<[u8; 32], F>,
) {
    x_index >>= 1;
    for oracle in tree {
        let mut p0 = x_index;
        let mut p1 = x_index ^ 1;
        if (p1 < p0) {
            p0 = x_index ^ 1;
            p1 = x_index;
        }
        if (oracle.len() == 1) {
            // 	    transcript.write_commitment(&oracle[0]);
            break;
        }
        transcript.write_commitment(&oracle[p0]);
        transcript.write_commitment(&oracle[p1]);
        x_index >>= 1;
    }
}

fn query_phase<F: PrimeField>(
    transcript: &mut impl TranscriptWrite<[u8; 32], F>,
    comm: &BasefoldCommitment<F>,
    oracles: &Vec<Type1Polynomial<F>>,
    num_verifier_queries: usize,
) -> (Vec<(Vec<(F, F)>, Vec<usize>)>, Vec<usize>) {
    let mut queries = transcript.squeeze_challenges(num_verifier_queries);

    let queries_usize: Vec<usize> = queries
        .iter()
        .map(|x_index| {
            let x_rep = (*x_index).into_bigint().to_bytes_be();
            let mut x: &[u8] = x_rep.as_ref();
            let (int_bytes, rest) = x.split_at(std::mem::size_of::<u32>());
            let x_int: u32 = u32::from_be_bytes(int_bytes.try_into().unwrap());
            ((x_int as usize) % comm.codeword.poly.len()).into()
        })
        .collect_vec();

    (
        queries_usize
            .par_iter()
            .map(|x_index| {
                return basefold_get_query::<F>(&comm.codeword, &oracles, *x_index);
            })
            .collect(),
        queries_usize,
    )
}

fn basefold_get_query<F: PrimeField>(
    first_oracle: &Type1Polynomial<F>,
    oracles: &Vec<Type1Polynomial<F>>,
    mut x_index: usize,
) -> (Vec<(F, F)>, Vec<usize>) {
    let mut queries = Vec::with_capacity(oracles.len() + 1);
    let mut indices = Vec::with_capacity(oracles.len() + 1);

    let mut p0 = x_index;
    let mut p1 = x_index ^ 1;

    if (p1 < p0) {
        p0 = x_index ^ 1;
        p1 = x_index;
    }
    queries.push((first_oracle.poly[p0], first_oracle.poly[p1]));
    indices.push(p0);
    x_index >>= 1;

    for oracle in oracles {
        let mut p0 = x_index;
        let mut p1 = x_index ^ 1;
        if (p1 < p0) {
            p0 = x_index ^ 1;
            p1 = x_index;
        }
        queries.push((oracle.poly[p0], oracle.poly[p1]));
        indices.push(p0);
        x_index >>= 1;
    }

    return (queries, indices);
}

fn commit_phase<F: PrimeField>(
    point: &[F],
    comm: &BasefoldCommitment<F>,
    transcript: &mut impl TranscriptWrite<[u8; 32], F>,
    num_vars: usize,
    num_rounds: usize,
    table_w_weights: &Vec<Vec<(F, F)>>,
    code_type: String,
    log_rate: usize,
) -> (
    Vec<Vec<Vec<[u8; 32]>>>,
    Vec<Vec<F>>,
    Vec<Type1Polynomial<F>>,
    Type1Polynomial<F>,
    Type1Polynomial<F>,
    F,
) {
    let mut oracles = Vec::with_capacity(num_vars);

    let mut trees = Vec::with_capacity(num_vars);

    let mut new_tree = &comm.codeword_tree;
    let mut root = new_tree[new_tree.len() - 1][0].clone();
    let mut new_oracle = &comm.codeword;

    let num_rounds = num_rounds;

    let mut eq = build_eq_x_r_vec::<F>(&point).unwrap();
    let mut eval = F::ZERO;
    let mut bh_evals = Type1Polynomial {
        poly: Vec::with_capacity(1 << num_vars),
    };
    for i in 0..eq.len() {
        eval = eval + comm.bh_evals.poly[i] * eq[i];
        bh_evals.poly.push(comm.bh_evals.poly[i]);
    }

    let mut eq = Type1Polynomial { poly: eq };
    let mut sum_check_oracles_vec = Vec::with_capacity(num_rounds + 1);
    let mut sum_check_oracle = sum_check_first_round::<F>(&mut eq, &mut bh_evals);
    sum_check_oracles_vec.push(sum_check_oracle.clone());

    for i in 0..(num_rounds) {
        transcript.write_commitment(&root).unwrap();
        transcript.write_field_elements(&sum_check_oracle);

        let challenge: F = transcript.squeeze_challenge();

        sum_check_oracle = sum_check_challenge_round(&mut eq, &mut bh_evals, challenge);

        sum_check_oracles_vec.push(sum_check_oracle.clone());

        oracles.push(basefold_one_round_by_interpolation_weights::<F>(
            &table_w_weights,
            i,
            new_oracle,
            challenge,
            num_vars,
            log_rate,
            code_type.clone(),
        ));

        new_oracle = &oracles[i];

        trees.push(merkelize::<F>(&new_oracle));

        root = trees[i][trees[i].len() - 1][0].clone();
    }
    transcript.write_field_elements(&sum_check_oracle);
    return (trees, sum_check_oracles_vec, oracles, bh_evals, eq, eval);
}

fn basefold_one_round_by_interpolation_weights<F: PrimeField>(
    table: &Vec<Vec<(F, F)>>,
    table_offset: usize,
    values: &Type1Polynomial<F>,
    challenge: F,
    num_vars: usize,
    log_rate: usize,
    code_type: String,
) -> Type1Polynomial<F> {
    let leveli = table.len() - 1 - table_offset;
    let level = &table[leveli];
    assert_eq!(1 << leveli, values.poly.len() >> 1);
    let fold = values
        .poly
        .chunks_exact(2)
        .enumerate()
        .map(|(i, ys)| {
            let mut x1 = F::ZERO;
            let mut x0 = F::ZERO;

            x0 = level[i].0;
            x1 = -x0;

            interpolate2_weights::<F>([(x0, ys[0]), (x1, ys[1])], level[i].1, challenge)
        })
        .collect::<Vec<_>>();
    Type1Polynomial { poly: fold }
}

pub fn interpolate2_weights<F: PrimeField>(points: [(F, F); 2], weight: F, x: F) -> F {
    // a0 -> a1
    // b0 -> b1
    // x  -> a1 + (x-a0)*(b1-a1)/(b0-a0)
    let (a0, a1) = points[0];
    let (b0, b1) = points[1];
    //    assert_ne!(a0, b0);
    a1 + (x - a0) * (b1 - a1) * weight
}

pub fn build_eq_x_r_vec<F: PrimeField>(r: &[F]) -> Option<Vec<F>> {
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    let mut eval = Vec::new();
    build_eq_x_r_helper(r, &mut eval);

    Some(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_helper<F: PrimeField>(r: &[F], buf: &mut Vec<F>) {
    assert!(!r.is_empty(), "r length is 0");

    if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push(F::ONE - r[0]);
        buf.push(r[0]);
    } else {
        build_eq_x_r_helper(&r[1..], buf);

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]
        // let mut res = vec![];
        // for &b_i in buf.iter() {
        //     let tmp = r[0] * b_i;
        //     res.push(b_i - tmp);
        //     res.push(tmp);
        // }
        // *buf = res;

        let mut res = vec![F::ZERO; buf.len() << 1];
        res.par_iter_mut().enumerate().for_each(|(i, val)| {
            let bi = buf[i >> 1];
            let tmp = r[0] * bi;
            if i & 1 == 0 {
                *val = bi - tmp;
            } else {
                *val = tmp;
            }
        });
        *buf = res;
    }
}

pub fn sum_check_first_round<F: PrimeField>(
    mut eq: &mut Type1Polynomial<F>,
    mut bh_values: &mut Type1Polynomial<F>,
) -> Vec<F> {
    one_level_interp_hc(&mut eq);
    one_level_interp_hc(&mut bh_values);
    parallel_pi(bh_values, eq)
}

pub fn one_level_interp_hc<F: PrimeField>(mut evals: &mut Type1Polynomial<F>) {
    if (evals.poly.len() == 1) {
        return;
    }
    evals.poly.par_chunks_mut(2).for_each(|chunk| {
        chunk[1] = chunk[1] - chunk[0];
    });
}

pub fn one_level_reverse_interp_hc<F: PrimeField>(mut evals: &mut Type1Polynomial<F>) {
    if (evals.poly.len() == 1) {
        return;
    }
    evals.poly.par_chunks_mut(2).for_each(|chunk| {
        chunk[1] = chunk[1] + chunk[0];
    });
}

pub fn one_level_eval_hc<F: PrimeField>(mut evals: &mut Type1Polynomial<F>, challenge: F) {
    evals.poly.par_chunks_mut(2).for_each(|chunk| {
        chunk[1] = chunk[0] + challenge * chunk[1];
    });
    let mut index = 0;

    evals.poly.retain(|v| {
        index += 1;
        (index - 1) % 2 == 1
    });
}

pub fn sum_check_challenge_round<F: PrimeField>(
    mut eq: &mut Type1Polynomial<F>,
    mut bh_values: &mut Type1Polynomial<F>,
    challenge: F,
) -> Vec<F> {
    one_level_eval_hc(&mut bh_values, challenge);
    one_level_eval_hc(&mut eq, challenge);

    one_level_interp_hc(&mut eq);
    one_level_interp_hc(&mut bh_values);

    parallel_pi(&bh_values, &eq)
    // p_i(&bh_values, &eq)
}

fn parallel_pi<F: PrimeField>(evals: &Type1Polynomial<F>, eq: &Type1Polynomial<F>) -> Vec<F> {
    if (evals.poly.len() == 1) {
        return vec![evals.poly[0], evals.poly[0], evals.poly[0]];
    }
    let mut coeffs = vec![F::ZERO, F::ZERO, F::ZERO];

    let mut firsts = vec![F::ZERO; evals.poly.len()];
    firsts.par_iter_mut().enumerate().for_each(|(i, mut f)| {
        if (i % 2 == 0) {
            *f = evals.poly[i] * eq.poly[i];
        }
    });

    let mut seconds = vec![F::ZERO; evals.poly.len()];
    seconds.par_iter_mut().enumerate().for_each(|(i, mut f)| {
        if (i % 2 == 0) {
            *f = evals.poly[i + 1] * eq.poly[i] + evals.poly[i] * eq.poly[i + 1];
        }
    });

    let mut thirds = vec![F::ZERO; evals.poly.len()];
    thirds.par_iter_mut().enumerate().for_each(|(i, mut f)| {
        if (i % 2 == 0) {
            *f = evals.poly[i + 1] * eq.poly[i + 1];
        }
    });

    coeffs[0] = firsts.par_iter().sum();
    coeffs[1] = seconds.par_iter().sum();
    coeffs[2] = thirds.par_iter().sum();

    coeffs
}

pub fn evaluate_over_foldable_domain<F: PrimeField>(
    log_rate: usize,
    mut coeffs: Type2Polynomial<F>,
    table: &Vec<Vec<F>>,
    code_type: String,
) -> Type1Polynomial<F> {
    // iterate over array, replacing even indices with (evals[i] - evals[(i+1)])
    let k = coeffs.poly.len();
    let logk = log2_strict(k);
    let cl = 1 << (logk + log_rate);
    let rate = 1 << log_rate;

    let mut coeffs_with_rep = vec![F::ZERO; cl];

    // base code - in this case is the repetition code

    for i in 0..k {
        for j in 0..rate {
            coeffs_with_rep[i * rate + j] = coeffs.poly[i];
        }
    }

    let mut chunk_size = rate; // block length of the base code
    for i in 0..logk {
        let level = &table[i + log_rate];
        chunk_size = chunk_size << 1; // k_j
        assert_eq!(level.len(), chunk_size >> 1);
        <Vec<F> as AsMut<[F]>>::as_mut(&mut coeffs_with_rep)
            .par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                let half_chunk = chunk_size >> 1;
                for j in half_chunk..chunk_size {
                    let rhs = chunk[j] * level[j - half_chunk];
                    let mut lhs = F::ZERO;
                    lhs = -rhs;
                    chunk[j] = chunk[j - half_chunk] + lhs;
                    chunk[j - half_chunk] = chunk[j - half_chunk] + rhs;
                }
            });
    }
    reverse_index_bits_in_place(&mut coeffs_with_rep);
    Type1Polynomial {
        poly: coeffs_with_rep,
    }
}

pub fn interpolate_over_boolean_hypercube_with_copy<F: PrimeField>(
    evals: &Type2Polynomial<F>,
) -> (Type2Polynomial<F>, Type1Polynomial<F>) {
    // iterate over array, replacing even indices with (evals[i] - evals[(i+1)])
    let n = log2_strict(evals.poly.len());
    let mut coeffs = vec![F::ZERO; evals.poly.len()];
    let mut new_evals = vec![F::ZERO; evals.poly.len()];

    let mut j = 0;
    while (j < coeffs.len()) {
        new_evals[j] = evals.poly[j];
        new_evals[j + 1] = evals.poly[j + 1];

        coeffs[j + 1] = evals.poly[j + 1] - evals.poly[j];
        coeffs[j] = evals.poly[j];
        j += 2
    }

    for i in 2..n + 1 {
        let chunk_size = 1 << i;
        coeffs.par_chunks_mut(chunk_size).for_each(|chunk| {
            let half_chunk = chunk_size >> 1;
            for j in half_chunk..chunk_size {
                chunk[j] = chunk[j] - chunk[j - half_chunk];
            }
        });
    }
    reverse_index_bits_in_place(&mut new_evals);
    (
        Type2Polynomial { poly: coeffs },
        Type1Polynomial { poly: new_evals },
    )
}

fn get_table_aes<F: PrimeField>(
    poly_size: usize,
    rate: usize,
    rng: &mut ChaCha8Rng,
) -> (Vec<Vec<(F, F)>>, Vec<Vec<F>>) {
    let lg_n: usize = rate + log2_strict(poly_size);

    let mut key: [u8; 16] = [0u8; 16];
    let mut iv: [u8; 16] = [0u8; 16];
    rng.fill_bytes(&mut key);
    rng.fill_bytes(&mut iv);

    type Aes128Ctr64LE = ctr::Ctr32LE<aes::Aes128>;

    let mut cipher = Aes128Ctr64LE::new(
        GenericArray::from_slice(&key[..]),
        GenericArray::from_slice(&iv[..]),
    );

    let bytes = (F::MODULUS_BIT_SIZE as usize).next_power_of_two() * (1 << lg_n) / 8;
    let mut dest: Vec<u8> = vec![0u8; bytes];
    cipher.apply_keystream(&mut dest[..]);

    let flat_table: Vec<F> = dest
        .par_chunks_exact((F::MODULUS_BIT_SIZE as usize).next_power_of_two() / 8)
        .map(|chunk| from_raw_bytes::<F>(&chunk.to_vec()))
        .collect::<Vec<_>>();

    assert_eq!(flat_table.len(), 1 << lg_n);

    let mut weights: Vec<F> = flat_table
        .par_iter()
        .map(|el| F::ZERO - *el - *el)
        .collect();

    // let mut scratch_space = vec![F::ZERO; weights.len()];
    // BatchInverter::invert_with_external_scratch(&mut weights, &mut
    // scratch_space);

    batch_inversion(&mut weights);

    let mut flat_table_w_weights = flat_table
        .iter()
        .zip(weights)
        .map(|(el, w)| (*el, w))
        .collect_vec();

    let mut unflattened_table_w_weights = vec![Vec::new(); lg_n];
    let mut unflattened_table = vec![Vec::new(); lg_n];

    let mut level_weights = flat_table_w_weights[0..2].to_vec();
    reverse_index_bits_in_place(&mut level_weights);
    unflattened_table_w_weights[0] = level_weights;

    unflattened_table[0] = flat_table[0..2].to_vec();
    for i in 1..lg_n {
        unflattened_table[i] = flat_table[(1 << i)..(1 << (i + 1))].to_vec();
        let mut level = flat_table_w_weights[(1 << i)..(1 << (i + 1))].to_vec();
        reverse_index_bits_in_place(&mut level);
        unflattened_table_w_weights[i] = level;
    }

    return (unflattened_table_w_weights, unflattened_table);
}

fn from_raw_bytes<F: PrimeField>(bytes: &Vec<u8>) -> F {
    let mut res = F::ZERO;
    bytes.into_iter().for_each(|b| {
        res += F::from(u64::from(*b));
    });
    res
}

// Ensure that SMALL_ARR_SIZE >= 4 * BIG_T_SIZE.
const BIG_T_SIZE: usize = 1 << 14;
const SMALL_ARR_SIZE: usize = 1 << 16;
pub fn reverse_index_bits_in_place<T>(arr: &mut [T]) {
    let n = arr.len();
    let lb_n = log2_strict(n);
    // If the whole array fits in fast cache, then the trivial algorithm is cache
    // friendly. Also, if `T` is really big, then the trivial algorithm is
    // cache-friendly, no matter the size of the array.
    if size_of::<T>() << lb_n <= SMALL_ARR_SIZE || size_of::<T>() >= BIG_T_SIZE {
        unsafe {
            reverse_index_bits_in_place_small(arr, lb_n);
        }
    } else {
        debug_assert!(n >= 4); // By our choice of `BIG_T_SIZE` and `SMALL_ARR_SIZE`.

        // Algorithm:
        //
        // Treat `arr` as a `sqrt(n)` by `sqrt(n)` row-major matrix. (Assume for now
        // that `lb_n` is even, i.e., `n` is a square number.) To perform
        // bit-order reversal we:
        //  1. Bit-reverse the order of the rows. (They are contiguous in memory, so
        //     this is basically a series of large `memcpy`s.)
        //  2. Transpose the matrix.
        //  3. Bit-reverse the order of the rows.
        // This is equivalent to, for every index `0 <= i < n`:
        //  1. bit-reversing `i[lb_n / 2..lb_n]`,
        //  2. swapping `i[0..lb_n / 2]` and `i[lb_n / 2..lb_n]`,
        //  3. bit-reversing `i[lb_n / 2..lb_n]`.
        //
        // If `lb_n` is odd, i.e., `n` is not a square number, then the above procedure
        // requires slight modification. At steps 1 and 3 we bit-reverse bits
        // `ceil(lb_n / 2)..lb_n`, of the index (shuffling `floor(lb_n / 2)`
        // chunks of length `ceil(lb_n / 2)`). At step 2, we perform _two_
        // transposes. We treat `arr` as two matrices, one where the middle bit of the
        // index is `0` and another, where the middle bit is `1`; we transpose each
        // individually.

        let lb_num_chunks = lb_n >> 1;
        let lb_chunk_size = lb_n - lb_num_chunks;
        unsafe {
            reverse_index_bits_in_place_chunks(arr, lb_num_chunks, lb_chunk_size);
            transpose_in_place_square(arr, lb_chunk_size, lb_num_chunks, 0);
            if lb_num_chunks != lb_chunk_size {
                // `arr` cannot be interpreted as a square matrix. We instead interpret it as a
                // `1 << lb_num_chunks` by `2` by `1 << lb_num_chunks` tensor, in row-major
                // order. The above transpose acted on `tensor[..., 0, ...]`
                // (all indices with middle bit `0`). We still need to transpose
                // `tensor[..., 1, ...]`. To do so, we advance arr by `1 <<
                // lb_num_chunks` effectively, adding that to every index.
                let arr_with_offset = &mut arr[1 << lb_num_chunks..];
                transpose_in_place_square(arr_with_offset, lb_chunk_size, lb_num_chunks, 0);
            }
            reverse_index_bits_in_place_chunks(arr, lb_num_chunks, lb_chunk_size);
        }
    }
}

/// Bit-reverse the order of elements in `arr`.
/// SAFETY: ensure that `arr.len() == 1 << lb_n`.
#[cfg(not(target_arch = "aarch64"))]
unsafe fn reverse_index_bits_in_place_small<T>(arr: &mut [T], lb_n: usize) {
    use super::BIT_REVERSE_6BIT;

    if lb_n <= 6 {
        // BIT_REVERSE_6BIT holds 6-bit reverses. This shift makes them lb_n-bit reverses.
        let dst_shr_amt = 6 - lb_n;
        for src in 0..arr.len() {
            let dst = (BIT_REVERSE_6BIT[src] as usize) >> dst_shr_amt;
            if src < dst {
                swap(arr.get_unchecked_mut(src), arr.get_unchecked_mut(dst));
            }
        }
    } else {
        // LLVM does not know that it does not need to reverse src at each iteration (which is
        // expensive on x86). We take advantage of the fact that the low bits of dst change rarely and the high
        // bits of dst are dependent only on the low bits of src.
        let dst_lo_shr_amt = 64 - (lb_n - 6);
        let dst_hi_shl_amt = lb_n - 6;
        for src_chunk in 0..(arr.len() >> 6) {
            let src_hi = src_chunk << 6;
            let dst_lo = src_chunk.reverse_bits() >> dst_lo_shr_amt;
            for src_lo in 0..(1 << 6) {
                let dst_hi = (BIT_REVERSE_6BIT[src_lo] as usize) << dst_hi_shl_amt;
                let src = src_hi + src_lo;
                let dst = dst_hi + dst_lo;
                if src < dst {
                    swap(arr.get_unchecked_mut(src), arr.get_unchecked_mut(dst));
                }
            }
        }
    }
}

const LB_BLOCK_SIZE: usize = 3;
#[cfg(target_arch = "aarch64")]
unsafe fn reverse_index_bits_in_place_small<T>(arr: &mut [T], lb_n: usize) {
    // Aarch64 can reverse bits in one instruction, so the trivial version works
    // best.

    use std::ptr::swap;

    for src in 0..arr.len() {
        // `wrapping_shr` handles the case when `arr.len() == 1`. In that case `src ==
        // 0`, so `src.reverse_bits() == 0`. `usize::wrapping_shr` by 64 is a
        // no-op, but it gives the correct result.
        let dst = src.reverse_bits().wrapping_shr(usize::BITS - lb_n as u32);
        if src < dst {
            swap(arr.get_unchecked_mut(src), arr.get_unchecked_mut(dst));
        }
    }
}

unsafe fn reverse_index_bits_in_place_chunks<T>(
    arr: &mut [T],
    lb_num_chunks: usize,
    lb_chunk_size: usize,
) {
    for i in 0..1usize << lb_num_chunks {
        // `wrapping_shr` handles the silly case when `lb_num_chunks == 0`.
        let j = i
            .reverse_bits()
            .wrapping_shr(usize::BITS - lb_num_chunks as u32);
        if i < j {
            swap_nonoverlapping(
                arr.get_unchecked_mut(i << lb_chunk_size),
                arr.get_unchecked_mut(j << lb_chunk_size),
                1 << lb_chunk_size,
            );
        }
    }
}

pub(crate) unsafe fn transpose_in_place_square<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
) {
    if lb_size <= LB_BLOCK_SIZE {
        transpose_in_place_square_small(arr, lb_stride, lb_size, x);
    } else {
        let lb_block_size = lb_size - 1;
        let block_size = 1 << lb_block_size;
        transpose_in_place_square(arr, lb_stride, lb_block_size, x);
        transpose_swap_square(arr, lb_stride, lb_block_size, x, x + block_size);
        transpose_in_place_square(arr, lb_stride, lb_block_size, x + block_size);
    }
}

unsafe fn transpose_in_place_square_small<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
) {
    for i in x + 1..x + (1 << lb_size) {
        for j in x..i {
            swap(
                arr.get_unchecked_mut(i + (j << lb_stride)),
                arr.get_unchecked_mut((i << lb_stride) + j),
            );
        }
    }
}

unsafe fn transpose_swap_square<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    y: usize,
) {
    if lb_size <= LB_BLOCK_SIZE {
        transpose_swap_square_small(arr, lb_stride, lb_size, x, y);
    } else {
        let lb_block_size = lb_size - 1;
        let block_size = 1 << lb_block_size;
        transpose_swap_square(arr, lb_stride, lb_block_size, x, y);
        transpose_swap_square(arr, lb_stride, lb_block_size, x + block_size, y);
        transpose_swap_square(arr, lb_stride, lb_block_size, x, y + block_size);
        transpose_swap_square(
            arr,
            lb_stride,
            lb_block_size,
            x + block_size,
            y + block_size,
        );
    }
}

unsafe fn transpose_swap_square_small<T>(
    arr: &mut [T],
    lb_stride: usize,
    lb_size: usize,
    x: usize,
    y: usize,
) {
    for i in x..x + (1 << lb_size) {
        for j in y..y + (1 << lb_size) {
            swap(
                arr.get_unchecked_mut(i + (j << lb_stride)),
                arr.get_unchecked_mut((i << lb_stride) + j),
            );
        }
    }
}

pub fn merkelize<F: PrimeField>(values: &Type1Polynomial<F>) -> Vec<Vec<[u8; 32]>> {
    let log_v = log2_strict(values.poly.len());
    let mut tree = Vec::with_capacity(log_v);
    let hashes = vec![Output::<Blake2s256>::default(); (values.poly.len() >> 1)];
    let res = hashes
        .par_iter()
        .enumerate()
        .map(|(i, mut hash)| {
            let mut hasher = Blake2s256::default();
            hasher.update(&values.poly[i + i].into_bigint().to_bytes_le());
            hasher.update(&values.poly[i + i + 1].into_bigint().to_bytes_le());
            // *hash = hasher.finalize();
            let array: [u8; 32] = hasher.finalize().into();
            array
        })
        .collect::<Vec<_>>();

    tree.push(res);

    for i in 1..(log_v) {
        let oracle = tree[i - 1]
            .par_chunks_exact(2)
            .map(|ys| {
                let mut hasher = Blake2s256::default();
                let mut hash: GenericArray<u8, _> = Output::<Blake2s256>::default();
                hasher.update(&ys[0]);
                hasher.update(&ys[1]);
                let array: [u8; 32] = hasher.finalize().into();
                array
            })
            .collect::<Vec<_>>();

        tree.push(oracle);
    }
    tree
}

#[derive(Debug)]
pub struct Ten {}
impl BasefoldExtParams for Ten {
    fn get_rate() -> usize {
        return 2;
    }

    fn get_basecode_rounds() -> usize {
        return 2;
    }

    fn get_reps() -> usize {
        return 1;
    }

    fn get_rs_basecode() -> bool {
        false
    }
    fn get_code_type() -> String {
        "random".to_string()
    }
}

#[test]
fn test_base_fold() {
    let k = 10;
    let timer = start_timer!(|| format!("PCS setup and trim -{k}"));
    let mut rng = ChaCha8Rng::from_entropy();
    let poly_size = 1 << k;
    let param = Basefold::<Fr, Ten>::setup(poly_size, 1, &mut rng).unwrap();

    let timer = start_timer!(|| format!("commit -{k}"));
    let mut transcript = Blake2s256Transcript::new(());

    let poly = Arc::new(DenseMultilinearExtension::<Fr>::rand(k, &mut rng));

    let (pp, vp) = Basefold::<Fr, Ten>::trim(&param, poly_size, 1).unwrap();

    let comm = Basefold::<Fr, Ten>::commit(&pp, &poly).unwrap();

    let point: Vec<Fr> = transcript.squeeze_challenges(k);

    let eval = poly.evaluate(&point).unwrap();

    // transcript.write_field_element(&eval).unwrap();
    Basefold::<Fr, Ten>::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();

    let proof = transcript.into_proof();

    let mut transcript = Blake2s256Transcript::from_proof((), proof.as_slice());
    let now = Instant::now();
    let b = Basefold::<Fr, Ten>::verify(
        &vp,
        // &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
        &point,
        &eval,
        &mut transcript,
        &mut rng,
    );
}

#[test]
fn test_transcript() {
    let k = 10;
    let timer = start_timer!(|| format!("PCS setup and trim -{k}"));
    let mut rng = ChaCha8Rng::from_entropy();
    let poly_size = 1 << k;
    let param = Basefold::<Fr, Ten>::setup(poly_size, 1, &mut rng).unwrap();

    let timer = start_timer!(|| format!("commit -{k}"));
    let mut transcript = Blake2s256Transcript::new(());

    let poly = Arc::new(DenseMultilinearExtension::<Fr>::rand(k, &mut rng));

    let (pp, vp) = Basefold::<Fr, Ten>::trim(&param, poly_size, 1).unwrap();

    let comm = Basefold::<Fr, Ten>::commit(&pp, &poly).unwrap();

    let new_tree = &comm.codeword_tree;
    let root = new_tree[new_tree.len() - 1][0].clone();

    <FiatShamirTranscript<std::io::Cursor<Vec<u8>>> as TranscriptWrite<[u8; 32], Fr>>::write_commitment(&mut transcript, &root).unwrap();
    let point: Vec<Fr> = transcript.squeeze_challenges(k);

    let eval = poly.evaluate(&point).unwrap();
    transcript.write_field_element(&eval).unwrap();

    let proof = transcript.into_proof();

    let mut transcript = Blake2s256Transcript::from_proof((), proof.as_slice());

    let res = <FiatShamirTranscript<std::io::Cursor<Vec<u8>>> as TranscriptRead<[u8; 32], Fr>>::read_commitment(&mut transcript).unwrap();

    // let res_eval = transcript.read_field_element().unwrap();

    assert_eq!(res, root);
}
