#![allow(warnings, unused)]
use self::{
    prover::{d_prove_zero_check, prove_perm_check},
    verifier::{d_verify_zero_check, verify_perm_check},
};
use crate::backend::hyperplonk::prover::d_prove_perm_check;
use crate::backend::hyperplonk::verifier::d_verify_perm_check;
use crate::{
    backend::{
        hyperplonk::{
            preprocessor::{batch_size, compose, permutation_polys},
            prover::{
                instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys,
                permutation_z_polys, prove_zero_check,
            },
            verifier::verify_zero_check,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    poly_iop::combined_check::CombinedCheckProof,
    util::{
        arithmetic::{powers, BooleanHypercube, PrimeField},
        end_timer,
        expression::Expression,
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    CombinedCheck, Error, PolyIOP,
};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet, Stats};
use rand::RngCore;
use std::time::Instant;
use std::{fmt::Debug, hash::Hash, iter, marker::PhantomData};
pub mod preprocessor;
pub(crate) mod prover;
pub(crate) mod verifier;

#[cfg(any(test, feature = "benchmark"))]
pub mod util;

#[derive(Clone, Debug)]
pub struct HyperPlonk<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::ProverParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    // pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub preprocess_polys: Vec<MultilinearPolynomial<F>>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
    pub(crate) permutation_comms: Vec<Pcs::Commitment>,
}

impl<F, Pcs> HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    pub fn new_to_file(
        pp: &HyperPlonkProverParam<F, Pcs>,
        file_path: &str,
    ) -> Result<(), std::io::Error> {
        // Handle file creation error with ?
        println!("path {} ", file_path);

        // 创建父目录（如果不存在）
        if let Some(parent) = std::path::Path::new(file_path).parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = std::fs::File::create(file_path)?;
        let writer = std::io::BufWriter::new(file);
        // Handle serialization error and convert to io::Error
        bincode::serialize_into(writer, &pp)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub fn read_from_file(file_path: &str) -> Result<Self, std::io::Error> {
        // Handle file open error with ?
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::new(file);
        // Handle deserialization error and convert to io::Error
        let setup = bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(setup)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::VerifierParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) num_lookups: usize,
    // pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_comms: Vec<(usize, Pcs::Commitment)>,
}

impl<F, Pcs> HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    pub fn new_to_file(
        pp: &HyperPlonkVerifierParam<F, Pcs>,
        file_path: &str,
    ) -> Result<(), std::io::Error> {
        println!("path {} ", file_path);

        // 创建父目录（如果不存在）
        if let Some(parent) = std::path::Path::new(file_path).parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let file = std::fs::File::create(file_path)?;
        let writer = std::io::BufWriter::new(file);
        // Handle serialization error and convert to io::Error
        bincode::serialize_into(writer, &pp)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub fn read_from_file(file_path: &str) -> Result<Self, std::io::Error> {
        // Handle file open error with ?
        let file = std::fs::File::open(file_path)?;
        let reader = std::io::BufReader::new(file);
        // Handle deserialization error and convert to io::Error
        let setup = bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(setup)
    }
}

impl<F, Pcs> PlonkishBackend<F> for HyperPlonk<Pcs>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned,
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    type Pcs = Pcs;
    type ProverParam = HyperPlonkProverParam<F, Pcs>;
    type VerifierParam = HyperPlonkVerifierParam<F, Pcs>;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<Pcs::Param, Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        Pcs::setup(poly_size, batch_size, rng)
    }

    fn preprocess_p(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(), Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;

        // Compute preprocesses comms
        let preprocess_polys = circuit_info
            .preprocess_polys
            .iter()
            .cloned()
            .map(MultilinearPolynomial::new)
            .collect_vec();

        let preprocess_comms = Pcs::batch_commit(&pcs_pp, &preprocess_polys)?;

        // Compute permutation polys and comms
        let permutation_polys = permutation_polys(
            num_vars,
            &circuit_info.permutation_polys(),
            &circuit_info.permutations,
        );
        let permutation_comms = Pcs::batch_commit(&pcs_pp, &permutation_polys)?;

        // Compose `VirtualPolynomialInfo`
        // let (num_permutation_z_polys, expression) = compose(circuit_info);
        let (expression) = compose(circuit_info);
        // let vp: Self::VerifierParam = HyperPlonkVerifierParam {
        //     pcs: pcs_vp,
        //     num_instances: circuit_info.num_instances.clone(),
        //     num_witness_polys: circuit_info.num_witness_polys.clone(),
        //     num_challenges: circuit_info.num_challenges.clone(),
        //     num_lookups: circuit_info.lookups.len(),
        //     num_permutation_z_polys,
        //     num_vars,
        //     expression: expression.clone(),
        //     preprocess_comms: preprocess_comms.clone(),
        //     permutation_comms: circuit_info
        //         .permutation_polys()
        //         .into_iter()
        //         .zip(permutation_comms.clone())
        //         .collect(),
        // };
        // let verifier_setup_filepath = format!("../data/Verifier-{}.paras", num_vars);
        // HyperPlonkVerifierParam::new_to_file(&vp, &verifier_setup_filepath);
        // drop(vp);
        let pp: Self::ProverParam = HyperPlonkProverParam {
            pcs: pcs_pp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            lookups: circuit_info.lookups.clone(),
            // num_permutation_z_polys,
            num_vars,
            expression,
            preprocess_polys,
            preprocess_comms,
            permutation_polys: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_polys)
                .collect(),
            permutation_comms,
        };
        let sub_prover_setup_filepath = format!("../data/SubProver-{}.paras", num_vars);
        HyperPlonkProverParam::new_to_file(&pp, &sub_prover_setup_filepath);
        Ok(())
    }

    fn preprocess_v(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(), Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;

        // Compute preprocesses comms
        let preprocess_polys = circuit_info
            .preprocess_polys
            .iter()
            .cloned()
            .map(MultilinearPolynomial::new)
            .collect_vec();

        let preprocess_comms = Pcs::batch_commit(&pcs_pp, &preprocess_polys)?;

        // Compute permutation polys and comms
        let permutation_polys = permutation_polys(
            num_vars,
            &circuit_info.permutation_polys(),
            &circuit_info.permutations,
        );
        let permutation_comms = Pcs::batch_commit(&pcs_pp, &permutation_polys)?;

        // Compose `VirtualPolynomialInfo`
        // let (num_permutation_z_polys, expression) = compose(circuit_info);
        let (expression) = compose(circuit_info);
        let vp: Self::VerifierParam = HyperPlonkVerifierParam {
            pcs: pcs_vp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            num_lookups: circuit_info.lookups.len(),
            // num_permutation_z_polys,
            num_vars,
            expression: expression,
            preprocess_comms: preprocess_comms,
            permutation_comms: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_comms)
                .collect(),
        };
        let verifier_setup_filepath = format!("../data/Verifier-{}.paras", num_vars);
        HyperPlonkVerifierParam::new_to_file(&vp, &verifier_setup_filepath);
        drop(vp);
        Ok(())
    }

    fn d_preprocess_p(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(), Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;

        // Compute preprocesses comms
        let preprocess_polys = circuit_info
            .preprocess_polys
            .iter()
            .cloned()
            .map(MultilinearPolynomial::new)
            .collect_vec();

        let preprocess_comms = Pcs::batch_commit(&pcs_pp, &preprocess_polys)?;

        // Compute permutation polys and comms
        let permutation_polys = permutation_polys(
            num_vars,
            &circuit_info.permutation_polys(),
            &circuit_info.permutations,
        );
        let permutation_comms = Pcs::batch_commit(&pcs_pp, &permutation_polys)?;

        // Compose `VirtualPolynomialInfo`
        let (expression) = compose(circuit_info);

        let pp: Self::ProverParam = HyperPlonkProverParam {
            pcs: pcs_pp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            lookups: circuit_info.lookups.clone(),
            // num_permutation_z_polys,
            num_vars,
            expression,
            preprocess_polys,
            preprocess_comms,
            permutation_polys: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_polys)
                .collect(),
            permutation_comms,
        };
        let sub_prover_setup_filepath =
            format!("../data/SubProver{}-{}.paras", Net::party_id(), num_vars);
        HyperPlonkProverParam::new_to_file(&pp, &sub_prover_setup_filepath);
        Ok(())
    }

    fn d_preprocess_v(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(), Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;

        // Compute preprocesses comms
        let preprocess_polys = circuit_info
            .preprocess_polys
            .iter()
            .cloned()
            .map(MultilinearPolynomial::new)
            .collect_vec();

        let preprocess_comms = Pcs::batch_commit(&pcs_pp, &preprocess_polys)?;

        // Compute permutation polys and comms
        let permutation_polys = permutation_polys(
            num_vars,
            &circuit_info.permutation_polys(),
            &circuit_info.permutations,
        );
        let permutation_comms = Pcs::batch_commit(&pcs_pp, &permutation_polys)?;

        // Compose `VirtualPolynomialInfo`
        // let (num_permutation_z_polys, expression) = compose(circuit_info);
        let (expression) = compose(circuit_info);
        let vp: Self::VerifierParam = HyperPlonkVerifierParam {
            pcs: pcs_vp,
            num_instances: circuit_info.num_instances.clone(),
            num_witness_polys: circuit_info.num_witness_polys.clone(),
            num_challenges: circuit_info.num_challenges.clone(),
            num_lookups: circuit_info.lookups.len(),
            // num_permutation_z_polys,
            num_vars,
            expression: expression,
            preprocess_comms: preprocess_comms,
            permutation_comms: circuit_info
                .permutation_polys()
                .into_iter()
                .zip(permutation_comms)
                .collect(),
        };
        let verifier_setup_filepath =
            format!("../data/Verifier{}-{}.paras", Net::party_id(), num_vars);
        HyperPlonkVerifierParam::new_to_file(&vp, &verifier_setup_filepath);
        drop(vp);
        Ok(())
    }

    fn prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys(pp.num_vars, instances)
        };

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>());
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)?
                .into_iter()
                .map(MultilinearPolynomial::new)
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = iter::empty()
            .chain(instance_polys.iter())
            .chain(pp.preprocess_polys.iter())
            .chain(witness_polys.iter())
            .collect_vec();

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let polys = iter::empty()
            .chain(polys)
            .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
            .collect_vec();
        challenges.extend([alpha]);
        let sumcheck_time = Instant::now();
        let (points, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        prove_perm_check(
            &iter::empty().chain(witness_polys.iter()).collect_vec(),
            &iter::empty().chain(witness_polys.iter()).collect_vec(),
            &iter::empty()
                .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
                .collect_vec(),
            transcript,
        );

        // PCS open

        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(pp.num_instances.len()))
            .chain(&pp.preprocess_comms)
            .chain(&witness_comms)
            .chain(&pp.permutation_comms)
            .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        let now = Instant::now();
        Pcs::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;

        // end_timer(timer);

        Ok(())
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 2);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // // Round n

        // let beta = transcript.squeeze_challenge();

        // let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // // Round n+1

        // let gamma = transcript.squeeze_challenge();

        // let lookup_h_permutation_z_comms = Pcs::read_commitments(
        //     &vp.pcs,
        //     vp.num_lookups + vp.num_permutation_z_polys,
        //     // vp.num_lookups,
        //     transcript,
        // )?;

        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        // challenges.extend([beta, gamma, alpha]);
        challenges.extend([alpha]);
        // challenges.extend([beta, alpha]);
        let (points, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        verify_perm_check(vp.num_vars, vp.num_witness_polys.iter().sum(), transcript);

        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(vp.num_instances.len()))
            .chain(&vp.preprocess_comms)
            .chain(&witness_comms)
            .chain(vp.permutation_comms.iter().map(|(_, comm)| comm))
            .collect_vec();
        Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;
        let mut waste = 0;
        while (transcript.read_commitment().is_ok()) {
            waste = waste + 1;
        }

        Ok(())
    }

    fn d_prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<crate::pcs::CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys(pp.num_vars, instances)
        };

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum());
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>());
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)?
                .into_iter()
                .map(MultilinearPolynomial::new)
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = iter::empty()
            .chain(instance_polys.iter())
            .chain(pp.preprocess_polys.iter())
            .chain(witness_polys.iter())
            .collect_vec();

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let polys = iter::empty()
            .chain(polys)
            .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
            .collect_vec();
        challenges.extend([alpha]);
        let sumcheck_time = Instant::now();
        let (points, evals) = d_prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        d_prove_perm_check(
            &iter::empty().chain(witness_polys.iter()).collect_vec(),
            &iter::empty().chain(witness_polys.iter()).collect_vec(),
            &iter::empty()
                .chain(pp.permutation_polys.iter().map(|(_, poly)| poly))
                .collect_vec(),
            transcript,
        );

        // PCS open

        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(pp.num_instances.len()))
            .chain(&pp.preprocess_comms)
            .chain(&witness_comms)
            .chain(&pp.permutation_comms)
            .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        let now = Instant::now();
        Pcs::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;

        // end_timer(timer);

        Ok(())
    }

    fn d_verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<crate::pcs::CommitmentChunk<F, Self::Pcs>, F>,
        rng: impl RngCore,
    ) -> Result<(), Error> {
        
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 2);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // // Round n

        // let beta = transcript.squeeze_challenge();

        // let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // // Round n+1

        // let gamma = transcript.squeeze_challenge();

        // let lookup_h_permutation_z_comms = Pcs::read_commitments(
        //     &vp.pcs,
        //     vp.num_lookups + vp.num_permutation_z_polys,
        //     // vp.num_lookups,
        //     transcript,
        // )?;

        // Round n+2

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        // challenges.extend([beta, gamma, alpha]);
        challenges.extend([alpha]);
        // challenges.extend([beta, alpha]);
        let (points, evals) = d_verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        d_verify_perm_check(vp.num_vars, vp.num_witness_polys.iter().sum(), transcript);

        let dummy_comm = Pcs::Commitment::default();
        let comms = iter::empty()
            .chain(iter::repeat(&dummy_comm).take(vp.num_instances.len()))
            .chain(&vp.preprocess_comms)
            .chain(&witness_comms)
            .chain(vp.permutation_comms.iter().map(|(_, comm)| comm))
            .collect_vec();
        Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;
        let mut waste = 0;
        while (transcript.read_commitment().is_ok()) {
            waste = waste + 1;
        }

        Ok(())
    }
}

impl<Pcs> WitnessEncoding for HyperPlonk<Pcs> {
    fn row_mapping(k: usize) -> Vec<usize> {
        BooleanHypercube::new(k).iter().skip(1).chain([0]).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{
            hyperplonk::{
                util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_with_lookup_circuit},
                HyperPlonk,
            },
            // test::run_plonkish_backend,
        },
        pcs::{
            multilinear::{
                Basefold, Gemini, MultilinearBrakedown, MultilinearHyrax, MultilinearIpa,
                MultilinearKzg, Zeromorph,
            },
            univariate::UnivariateKzg,
        },
        util::{
            code::BrakedownSpec6, hash::Keccak256, test::seeded_std_rng,
            transcript::Keccak256Transcript,
        },
    };
    use halo2_curves::{
        bn256::{self, Bn256},
        grumpkin,
    };

    // macro_rules! tests {
    //     ($name:ident, $pcs:ty, $num_vars_range:expr) => {
    //         paste::paste! {
    //             #[test]
    //             fn [<$name _hyperplonk_vanilla_plonk>]() {
    //                 run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
    //                     rand_vanilla_plonk_circuit(num_vars, seeded_std_rng(), seeded_std_rng())
    //                 });
    //             }

    //             #[test]
    //             fn [<$name _hyperplonk_vanilla_plonk_with_lookup>]() {
    //                 run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
    //                     rand_vanilla_plonk_with_lookup_circuit(num_vars, seeded_std_rng(), seeded_std_rng())
    //                 });
    //             }
    //         }
    //     };
    //     ($name:ident, $pcs:ty) => {
    //         tests!($name, $pcs, 2..16);
    //     };
    // }

    //    tests!(basefold, Basefold<bn256::Fr, Keccak256>);
    //    tests!(brakedown, MultilinearBrakedown<bn256::Fr, Keccak256, BrakedownSpec6>);
    //    tests!(hyrax, MultilinearHyrax<grumpkin::G1Affine>, 5..16);
    //    tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    //    tests!(kzg, MultilinearKzg<Bn256>);
    //    tests!(gemini_kzg, Gemini<UnivariateKzg<Bn256>>);
    //    tests!(zeromorph_kzg, Zeromorph<UnivariateKzg<Bn256>>);
}
