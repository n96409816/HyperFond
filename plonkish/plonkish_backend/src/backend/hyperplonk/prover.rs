use super::util::computer_nums_and_denoms;
use crate::Math;
use crate::{
    backend::{
        hyperplonk::{
            verifier::{pcs_query, point_offset, points},
            HyperPlonk,
        },
        WitnessEncoding,
    },
    pcs::Evaluation,
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::{multilinear::MultilinearPolynomial, Polynomial},
    util::{
        arithmetic::{
            div_ceil, inner_product, steps_by, sum, BatchInvert, BooleanHypercube, PrimeField,
        },
        end_timer,
        expression::{CommonPolynomial, Expression, Query, Rotation},
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        start_timer,
        transcript::FieldTranscriptWrite,
        Itertools,
    },
    Error, PolyIOPErrors,
};
use ark_std::end_timer;
use ark_std::start_timer;
use itertools::izip;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    iter,
};

pub(crate) fn instance_polys<'a, F: PrimeField>(
    num_vars: usize,
    instances: impl IntoIterator<Item = impl IntoIterator<Item = &'a F>>,
) -> Vec<MultilinearPolynomial<F>> {
    let row_mapping = HyperPlonk::<()>::row_mapping(num_vars);
    instances
        .into_iter()
        .map(|instances| {
            let mut poly = vec![F::ZERO; 1 << num_vars];
            for (b, instance) in row_mapping.iter().zip(instances.into_iter()) {
                poly[*b] = *instance;
            }
            poly
        })
        .map(MultilinearPolynomial::new)
        .collect()
}

pub(crate) fn lookup_compressed_polys<F: PrimeField>(
    lookups: &[Vec<(Expression<F>, Expression<F>)>],
    polys: &[&MultilinearPolynomial<F>],
    challenges: &[F],
    betas: &[F],
) -> Vec<[MultilinearPolynomial<F>; 2]> {
    if lookups.is_empty() {
        return Default::default();
    }

    let num_vars = polys[0].num_vars();
    let expression = lookups
        .iter()
        .flat_map(|lookup| lookup.iter().map(|(input, table)| (input + table)))
        .sum::<Expression<_>>();
    let lagranges = {
        let bh = BooleanHypercube::new(num_vars).iter().collect_vec();
        expression
            .used_langrange()
            .into_iter()
            .map(|i| (i, bh[i.rem_euclid(1 << num_vars) as usize]))
            .collect::<HashSet<_>>()
    };
    lookups
        .iter()
        .map(|lookup| lookup_compressed_poly(lookup, &lagranges, polys, challenges, betas))
        .collect()
}

pub(super) fn lookup_compressed_poly<F: PrimeField>(
    lookup: &[(Expression<F>, Expression<F>)],
    lagranges: &HashSet<(i32, usize)>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: &[F],
    betas: &[F],
) -> [MultilinearPolynomial<F>; 2] {
    let num_vars = polys[0].num_vars();
    let bh = BooleanHypercube::new(num_vars);
    let compress = |expressions: &[&Expression<F>]| {
        betas
            .iter()
            .copied()
            .zip(expressions.iter().map(|expression| {
                let mut compressed = vec![F::ZERO; 1 << num_vars];
                parallelize(&mut compressed, |(compressed, start)| {
                    for (b, compressed) in (start..).zip(compressed) {
                        *compressed = expression.evaluate(
                            &|constant| constant,
                            &|common_poly| match common_poly {
                                CommonPolynomial::Identity => F::from(b as u64),
                                CommonPolynomial::Lagrange(i) => {
                                    if lagranges.contains(&(i, b)) {
                                        F::ONE
                                    } else {
                                        F::ZERO
                                    }
                                }
                                CommonPolynomial::EqXY(_) => unreachable!(),
                            },
                            &|query| polys[query.poly()][bh.rotate(b, query.rotation())],
                            &|challenge| challenges[challenge],
                            &|value| -value,
                            &|lhs, rhs| lhs + &rhs,
                            &|lhs, rhs| lhs * &rhs,
                            &|value, scalar| value * &scalar,
                        );
                    }
                });
                MultilinearPolynomial::new(compressed)
            }))
            .sum::<MultilinearPolynomial<_>>()
    };

    let (inputs, tables) = lookup
        .iter()
        .map(|(input, table)| (input, table))
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let timer = start_timer(|| "compressed_input_poly");
    let compressed_input_poly = compress(&inputs);
    end_timer(timer);

    let timer = start_timer(|| "compressed_table_poly");
    let compressed_table_poly = compress(&tables);
    end_timer(timer);

    [compressed_input_poly, compressed_table_poly]
}

pub(crate) fn lookup_m_polys<F: PrimeField + Hash>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
) -> Result<Vec<MultilinearPolynomial<F>>, Error> {
    compressed_polys.iter().map(lookup_m_poly).try_collect()
}

pub(super) fn lookup_m_poly<F: PrimeField + Hash>(
    compressed_polys: &[MultilinearPolynomial<F>; 2],
) -> Result<MultilinearPolynomial<F>, Error> {
    let [input, table] = compressed_polys;

    let counts = {
        let indice_map = table.iter().zip(0..).collect::<HashMap<_, usize>>();

        let chunk_size = div_ceil(input.evals().len(), num_threads());
        let num_chunks = div_ceil(input.evals().len(), chunk_size);
        let mut counts = vec![HashMap::new(); num_chunks];
        let mut valids = vec![true; num_chunks];
        parallelize_iter(
            counts
                .iter_mut()
                .zip(valids.iter_mut())
                .zip((0..).step_by(chunk_size)),
            |((count, valid), start)| {
                for input in input[start..].iter().take(chunk_size) {
                    if let Some(idx) = indice_map.get(input) {
                        count
                            .entry(*idx)
                            .and_modify(|count| *count += 1)
                            .or_insert(1);
                    } else {
                        *valid = false;
                        break;
                    }
                }
            },
        );
        if valids.iter().any(|valid| !valid) {
            return Err(Error::InvalidSnark("Invalid lookup input".to_string()));
        }
        counts
    };

    let mut m = vec![0; 1 << input.num_vars()];
    for (idx, count) in counts.into_iter().flatten() {
        m[idx] += count;
    }
    let m = par_map_collect(m, |count| match count {
        0 => F::ZERO,
        1 => F::ONE,
        count => F::from(count),
    });
    Ok(MultilinearPolynomial::new(m))
}

pub(super) fn lookup_h_polys<F: PrimeField + Hash>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
    m_polys: &[MultilinearPolynomial<F>],
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    compressed_polys
        .iter()
        .zip(m_polys.iter())
        .map(|(compressed_polys, m_poly)| lookup_h_poly(compressed_polys, m_poly, gamma))
        .collect()
}

pub(super) fn lookup_h_poly<F: PrimeField + Hash>(
    compressed_polys: &[MultilinearPolynomial<F>; 2],
    m_poly: &MultilinearPolynomial<F>,
    gamma: &F,
) -> MultilinearPolynomial<F> {
    let [input, table] = compressed_polys;
    let mut h_input = vec![F::ZERO; 1 << input.num_vars()];
    let mut h_table = vec![F::ZERO; 1 << input.num_vars()];

    parallelize(&mut h_input, |(h_input, start)| {
        for (h_input, input) in h_input.iter_mut().zip(input[start..].iter()) {
            *h_input = *gamma + input;
        }
    });
    parallelize(&mut h_table, |(h_table, start)| {
        for (h_table, table) in h_table.iter_mut().zip(table[start..].iter()) {
            *h_table = *gamma + table;
        }
    });

    let chunk_size = div_ceil(2 * h_input.len(), num_threads());
    parallelize_iter(
        iter::empty()
            .chain(h_input.chunks_mut(chunk_size))
            .chain(h_table.chunks_mut(chunk_size)),
        |h| {
            h.iter_mut().batch_invert();
        },
    );

    parallelize(&mut h_input, |(h_input, start)| {
        for (h_input, (h_table, m)) in h_input
            .iter_mut()
            .zip(h_table[start..].iter().zip(m_poly[start..].iter()))
        {
            *h_input -= *h_table * m;
        }
    });

    if cfg!(feature = "sanity-check") {
        assert_eq!(sum::<F>(&h_input), F::ZERO);
    }

    MultilinearPolynomial::new(h_input)
}

pub(crate) fn permutation_z_polys<F: PrimeField>(
    num_chunks: usize,
    permutation_polys: &[(usize, MultilinearPolynomial<F>)],
    polys: &[&MultilinearPolynomial<F>],
    beta: &F,
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    if permutation_polys.is_empty() {
        return Vec::new();
    }

    let chunk_size = div_ceil(permutation_polys.len(), num_chunks);
    let num_vars = polys[0].num_vars();

    let timer = start_timer(|| "products");
    let products = permutation_polys
        .chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, permutation_polys)| {
            let mut product = vec![F::ONE; 1 << num_vars];

            for (poly, permutation_poly) in permutation_polys.iter() {
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), permutation) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(permutation_poly[start..].iter())
                    {
                        *product *= (*beta * permutation) + gamma + value;
                    }
                });
            }

            parallelize(&mut product, |(product, _)| {
                product.iter_mut().batch_invert();
            });

            for ((poly, _), idx) in permutation_polys.iter().zip(chunk_idx * chunk_size..) {
                let id_offset = idx << num_vars;
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), beta_id) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(steps_by(F::from((id_offset + start) as u64) * beta, *beta))
                    {
                        *product *= beta_id + gamma + value;
                    }
                });
            }

            product
        })
        .collect_vec();
    end_timer(timer);

    let timer = start_timer(|| "z_polys");
    let z = iter::empty()
        .chain(iter::repeat(F::ZERO).take(num_chunks))
        .chain(Some(F::ONE))
        .chain(
            BooleanHypercube::new(num_vars)
                .iter()
                .skip(1)
                .flat_map(|b| iter::repeat(b).take(num_chunks))
                .zip(products.iter().cycle())
                .scan(F::ONE, |state, (b, product)| {
                    *state *= &product[b];
                    Some(*state)
                }),
        )
        .take(num_chunks << num_vars)
        .collect_vec();

    if cfg!(feature = "sanity-check") {
        let b_last = BooleanHypercube::new(num_vars).iter().last().unwrap();
        assert_eq!(
            *z.last().unwrap() * products.last().unwrap()[b_last],
            F::ONE
        );
    }

    drop(products);
    end_timer(timer);

    let _timer = start_timer(|| "into_bh_order");
    let nth_map = BooleanHypercube::new(num_vars)
        .nth_map()
        .into_iter()
        .map(|b| num_chunks * b)
        .collect_vec();
    (0..num_chunks)
        .map(|offset| MultilinearPolynomial::new(par_map_collect(&nth_map, |b| z[offset + b])))
        .collect()
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_zero_check<F: PrimeField + Serialize + DeserializeOwned>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    prove_sum_check(
        num_instance_poly,
        expression,
        F::ZERO,
        polys,
        challenges,
        y,
        transcript,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn d_prove_zero_check<F: PrimeField + Serialize + DeserializeOwned>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    d_prove_sum_check(
        num_instance_poly,
        expression,
        F::ZERO,
        polys,
        challenges,
        y,
        transcript,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_perm_check<F: PrimeField + Serialize + DeserializeOwned>(
    fxs: &[&MultilinearPolynomial<F>],
    gxs: &[&MultilinearPolynomial<F>],
    perms: &[&MultilinearPolynomial<F>],
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(), PolyIOPErrors> {
    let start = start_timer!(|| "Permutation check prove");
    if fxs.is_empty() {
        return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
    }
    if (fxs.len() != gxs.len()) || (fxs.len() != perms.len()) {
        return Err(PolyIOPErrors::InvalidProof(format!(
            "fxs.len() = {}, gxs.len() = {}, perms.len() = {}",
            fxs.len(),
            gxs.len(),
            perms.len(),
        )));
    }

    let num_vars = fxs[0].num_vars;
    for ((fx, gx), perm) in fxs.iter().zip(gxs.iter()).zip(perms.iter()) {
        if (fx.num_vars != num_vars) || (gx.num_vars != num_vars) || (perm.num_vars != num_vars) {
            return Err(PolyIOPErrors::InvalidParameters(
                "number of variables unmatched".to_string(),
            ));
        }
    }
    // println!("fxs.len(){:?}", fxs.len());
    // generate challenge `beta` and `gamma` from current transcript
    let beta = transcript.squeeze_challenge();
    let gamma = transcript.squeeze_challenge();

    let num_points = 1 << num_vars;
    let px_evals: Vec<F> = (0..num_points)
        .flat_map(|_| [-F::ONE, F::ONE]) // -1 for y=0, 1 for y=1
        .collect();

    for (fx, gx, perm) in izip!(fxs, gxs, perms) {
        let (lefts, rights) = computer_nums_and_denoms(&beta, &gamma, fx, gx, perm)?;

        // Build qx directly using iterators
        let qx_evals: Vec<F> = lefts
            .evals
            .iter()
            .zip(&rights.evals)
            .flat_map(|(l, r)| [*l, *r])
            .collect();

        // Reuse precomputed px_evals
        let px_poly = MultilinearPolynomial::from_evals(px_evals.clone());
        let qx_poly = MultilinearPolynomial::from_evals(qx_evals);

        prove_fractional_sum_check(&px_poly, &qx_poly, transcript)?;
    }

    end_timer!(start);
    Ok(())
}

#[allow(clippy::type_complexity)]
pub(super) fn d_prove_perm_check<F: PrimeField + Serialize + DeserializeOwned>(
    fxs: &[&MultilinearPolynomial<F>],
    gxs: &[&MultilinearPolynomial<F>],
    perms: &[&MultilinearPolynomial<F>],
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(), PolyIOPErrors> {
    let start = start_timer!(|| "Permutation check prove");
    if fxs.is_empty() {
        return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
    }
    if (fxs.len() != gxs.len()) || (fxs.len() != perms.len()) {
        return Err(PolyIOPErrors::InvalidProof(format!(
            "fxs.len() = {}, gxs.len() = {}, perms.len() = {}",
            fxs.len(),
            gxs.len(),
            perms.len(),
        )));
    }

    let num_vars = fxs[0].num_vars;
    for ((fx, gx), perm) in fxs.iter().zip(gxs.iter()).zip(perms.iter()) {
        if (fx.num_vars != num_vars) || (gx.num_vars != num_vars) || (perm.num_vars != num_vars) {
            return Err(PolyIOPErrors::InvalidParameters(
                "number of variables unmatched".to_string(),
            ));
        }
    }
    // println!("fxs.len(){:?}", fxs.len());
    // generate challenge `beta` and `gamma` from current transcript
    let beta = transcript.squeeze_challenge();
    let gamma = transcript.squeeze_challenge();

    let num_points = 1 << num_vars;
    let px_evals: Vec<F> = (0..num_points)
        .flat_map(|_| [-F::ONE, F::ONE]) // -1 for y=0, 1 for y=1
        .collect();

    for (fx, gx, perm) in izip!(fxs, gxs, perms) {
        let (lefts, rights) = computer_nums_and_denoms(&beta, &gamma, fx, gx, perm)?;

        // Build qx directly using iterators
        let qx_evals: Vec<F> = lefts
            .evals
            .iter()
            .zip(&rights.evals)
            .flat_map(|(l, r)| [*l, *r])
            .collect();

        // Reuse precomputed px_evals
        let px_poly = MultilinearPolynomial::from_evals(px_evals.clone());
        let qx_poly = MultilinearPolynomial::from_evals(qx_evals);

        d_prove_fractional_sum_check(&px_poly, &qx_poly, transcript)?;
    }

    end_timer!(start);
    Ok(())
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_fractional_sum_check<F: PrimeField + Serialize + DeserializeOwned>(
    px: &MultilinearPolynomial<F>,
    qx: &MultilinearPolynomial<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(), PolyIOPErrors> {
    let start = start_timer!(|| "Fractional sumcheck prove");

    if px.evals.len() != qx.evals.len() {
        return Err(PolyIOPErrors::InvalidProof(format!(
            "pxs.len() = {}, qxs.len() = {}",
            px.evals.len(),
            qx.evals.len(),
        )));
    }

    let num_vars = px.num_vars;
    let mut hatpx = Vec::with_capacity(num_vars);
    let mut hatqx = Vec::with_capacity(num_vars);

    // Use slices instead of owned vectors for iterative computation
    let mut current_p = &px.evals[..];
    let mut current_q = &qx.evals[..];

    for _ in 0..num_vars {
        let chunk_size = current_p.len() / 2;
        if chunk_size == 0 {
            break;
        }

        let mut hatpxi = Vec::with_capacity(chunk_size);
        let mut hatqxi = Vec::with_capacity(chunk_size);

        for (chunk_p, chunk_q) in current_p.chunks_exact(2).zip(current_q.chunks_exact(2)) {
            hatpxi.push(chunk_p[0] * chunk_q[1] + chunk_p[1] * chunk_q[0]);
            hatqxi.push(chunk_q[0] * chunk_q[1]);
        }

        hatpx.push(MultilinearPolynomial::from_evals(hatpxi.clone()));
        hatqx.push(MultilinearPolynomial::from_evals(hatqxi.clone()));

        current_p = &hatpx.last().unwrap().evals;
        current_q = &hatqx.last().unwrap().evals;
    }

    // Process intermediate levels
    for i in 2..num_vars {
        let idx = num_vars - 1 - i;
        let poly_p = &hatpx[idx];
        let poly_q = &hatqx[idx];
        let vars = poly_p.num_vars;

        let r = transcript.squeeze_challenges(vars);
        let lamda = transcript.squeeze_challenge();
        let eq_xt = MultilinearPolynomial::<F>::eq_xy(&r);
        let eq_xt_evals = &eq_xt.evals[..poly_p.evals.len()];

        // Compute inner products directly without intermediate vector
        let inner_p = inner_product(&poly_p.evals, eq_xt_evals);
        let inner_q = inner_product(&poly_q.evals, eq_xt_evals);
        let tilde_gs_sum = lamda * inner_p + inner_q;

        transcript.write_field_element(&tilde_gs_sum);

        // Reuse polynomials without cloning
        let expression = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()))
            * Expression::<F>::eq_xy(0)
            + Expression::<F>::Polynomial(Query::new(1, Rotation::cur()))
                * Expression::<F>::eq_xy(0);

        let polys = vec![poly_p.clone(), poly_q.clone()];
        let binding = [r];
        let challenges = [lamda];

        let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &binding);

        ClassicSumCheck::<EvaluationsProver<_>>::prove(
            &(),
            vars,
            virtual_poly,
            tilde_gs_sum,
            transcript,
        );
    }

    // Final processing
    let vars = px.num_vars;
    let r = transcript.squeeze_challenges(vars);
    let lamda = transcript.squeeze_challenge();
    let eq_xt = MultilinearPolynomial::<F>::eq_xy(&r);
    let eq_xt_evals = &eq_xt.evals[..px.evals.len()];

    // Compute inner products directly
    let inner_p = inner_product(&px.evals, eq_xt_evals);
    let inner_q = inner_product(&qx.evals, eq_xt_evals);
    let tilde_gs_sum = lamda * inner_p + inner_q;

    transcript.write_field_element(&tilde_gs_sum);

    let expression = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()))
        * Expression::<F>::eq_xy(0)
        + Expression::<F>::Polynomial(Query::new(1, Rotation::cur())) * Expression::<F>::eq_xy(0);

    let polys = vec![px.clone(), qx.clone()];
    let binding = [r];
    let challenges = [lamda];

    let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &binding);

    ClassicSumCheck::<EvaluationsProver<_>>::prove(
        &(),
        vars,
        virtual_poly,
        tilde_gs_sum,
        transcript,
    );

    end_timer!(start);
    Ok(())
}

#[allow(clippy::type_complexity)]
pub(super) fn d_prove_fractional_sum_check<F: PrimeField + Serialize + DeserializeOwned>(
    px: &MultilinearPolynomial<F>,
    qx: &MultilinearPolynomial<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(), PolyIOPErrors> {
    let start = start_timer!(|| "Fractional sumcheck prove");

    if px.evals.len() != qx.evals.len() {
        return Err(PolyIOPErrors::InvalidProof(format!(
            "pxs.len() = {}, qxs.len() = {}",
            px.evals.len(),
            qx.evals.len(),
        )));
    }

    let num_vars = px.num_vars;
    let mut hatpx = Vec::with_capacity(num_vars);
    let mut hatqx = Vec::with_capacity(num_vars);

    // Use slices instead of owned vectors for iterative computation
    let mut current_p = &px.evals[..];
    let mut current_q = &qx.evals[..];

    for _ in 0..num_vars {
        let chunk_size = current_p.len() / 2;
        if chunk_size == 0 {
            break;
        }

        let mut hatpxi = Vec::with_capacity(chunk_size);
        let mut hatqxi = Vec::with_capacity(chunk_size);

        for (chunk_p, chunk_q) in current_p.chunks_exact(2).zip(current_q.chunks_exact(2)) {
            hatpxi.push(chunk_p[0] * chunk_q[1] + chunk_p[1] * chunk_q[0]);
            hatqxi.push(chunk_q[0] * chunk_q[1]);
        }

        hatpx.push(MultilinearPolynomial::from_evals(hatpxi.clone()));
        hatqx.push(MultilinearPolynomial::from_evals(hatqxi.clone()));

        current_p = &hatpx.last().unwrap().evals;
        current_q = &hatqx.last().unwrap().evals;
    }

    // Process intermediate levels
    for i in 2..num_vars {
        let idx = num_vars - 1 - i;
        let poly_p = &hatpx[idx];
        let poly_q = &hatqx[idx];
        let vars = poly_p.num_vars;

        let r = transcript.squeeze_challenges(vars);
        let lamda = transcript.squeeze_challenge();
        let eq_xt = MultilinearPolynomial::<F>::eq_xy(&r);
        let eq_xt_evals = &eq_xt.evals[..poly_p.evals.len()];

        // Compute inner products directly without intermediate vector
        let inner_p = inner_product(&poly_p.evals, eq_xt_evals);
        let inner_q = inner_product(&poly_q.evals, eq_xt_evals);
        let tilde_gs_sum = lamda * inner_p + inner_q;

        transcript.write_field_element(&tilde_gs_sum);

        // Reuse polynomials without cloning
        let expression = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()))
            * Expression::<F>::eq_xy(0)
            + Expression::<F>::Polynomial(Query::new(1, Rotation::cur()))
                * Expression::<F>::eq_xy(0);

        let polys = vec![poly_p.clone(), poly_q.clone()];
        let binding = [r];
        let challenges = [lamda];

        let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &binding);

        ClassicSumCheck::<EvaluationsProver<_>>::d_prove(
            &(),
            vars,
            virtual_poly,
            tilde_gs_sum,
            transcript,
        );
    }

    // Final processing
    let vars = px.num_vars;
    let r = transcript.squeeze_challenges(vars);
    let lamda = transcript.squeeze_challenge();
    let eq_xt = MultilinearPolynomial::<F>::eq_xy(&r);
    let eq_xt_evals = &eq_xt.evals[..px.evals.len()];

    // Compute inner products directly
    let inner_p = inner_product(&px.evals, eq_xt_evals);
    let inner_q = inner_product(&qx.evals, eq_xt_evals);
    let tilde_gs_sum = lamda * inner_p + inner_q;

    transcript.write_field_element(&tilde_gs_sum);

    let expression = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()))
        * Expression::<F>::eq_xy(0)
        + Expression::<F>::Polynomial(Query::new(1, Rotation::cur())) * Expression::<F>::eq_xy(0);

    let polys = vec![px.clone(), qx.clone()];
    let binding = [r];
    let challenges = [lamda];

    let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &binding);

    ClassicSumCheck::<EvaluationsProver<_>>::d_prove(
        &(),
        vars,
        virtual_poly,
        tilde_gs_sum,
        transcript,
    );

    end_timer!(start);
    Ok(())
}

#[allow(clippy::type_complexity)]
pub(crate) fn prove_sum_check<F: PrimeField + Serialize + DeserializeOwned>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    sum: F,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    let num_vars = polys[0].num_vars();
    let ys = [y];
    let virtual_poly = VirtualPolynomial::new(expression, polys.to_vec(), &challenges, &ys);
    let (x, evals) = ClassicSumCheck::<EvaluationsProver<_>>::prove(
        &(),
        num_vars,
        virtual_poly,
        sum,
        transcript,
    )?;

    let pcs_query = pcs_query(expression, num_instance_poly);
    let point_offset = point_offset(&pcs_query);

    let timer = start_timer(|| format!("evals-{}", pcs_query.len()));
    let evals = pcs_query
        .iter()
        .flat_map(|query| {
            (point_offset[&query.rotation()]..)
                .zip(if query.rotation() == Rotation::cur() {
                    vec![evals[query.poly()]]
                } else {
                    polys[query.poly()].evaluate_for_rotation(&x, query.rotation())
                })
                .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
        })
        .collect_vec();
    end_timer(timer);

    transcript.write_field_elements(evals.iter().map(Evaluation::value))?;

    Ok((points(&pcs_query, &x), evals))
}

#[allow(clippy::type_complexity)]
pub(crate) fn d_prove_sum_check<F: PrimeField + Serialize + DeserializeOwned>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    sum: F,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    let num_vars = polys[0].num_vars();
    let ys = [y];
    let virtual_poly = VirtualPolynomial::new(expression, polys.to_vec(), &challenges, &ys);
    let (x, evals) = ClassicSumCheck::<EvaluationsProver<_>>::d_prove(
        &(),
        num_vars,
        virtual_poly,
        sum,
        transcript,
    )?;

    let pcs_query = pcs_query(expression, num_instance_poly);
    let point_offset = point_offset(&pcs_query);

    let timer = start_timer(|| format!("evals-{}", pcs_query.len()));
    let evals = pcs_query
        .iter()
        .flat_map(|query| {
            (point_offset[&query.rotation()]..)
                .zip(if query.rotation() == Rotation::cur() {
                    vec![evals[query.poly()]]
                } else {
                    polys[query.poly()].evaluate_for_rotation(&x, query.rotation())
                })
                .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
        })
        .collect_vec();
    end_timer(timer);

    transcript.write_field_elements(evals.iter().map(Evaluation::value))?;

    Ok((points(&pcs_query, &x), evals))
}
