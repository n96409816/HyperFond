// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

pub mod dbasefold;
pub mod errors;
pub mod prelude;

use ark_ec::pairing::Pairing;
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;
use std::{borrow::Borrow, fmt::Debug, hash::Hash};

use crate::util::transcript::{self, TranscriptRead, TranscriptWrite};

use self::errors::PCSError;

/// API definitions for structured reference string
pub trait StructuredReferenceString<E: Pairing>: Sized {
    /// Prover parameters
    type ProverParam;
    /// Verifier parameters
    type VerifierParam;

    /// Extract the prover parameters from the public parameters.
    fn extract_prover_param(&self, supported_size: usize) -> Self::ProverParam;
    /// Extract the verifier parameters from the public parameters.
    fn extract_verifier_param(&self, supported_size: usize) -> Self::VerifierParam;

    /// Trim the universal parameters to specialize the public parameters
    /// for polynomials to the given `supported_size`, and
    /// returns committer key and verifier key.
    ///
    /// - For univariate polynomials, `supported_size` is the maximum degree.
    /// - For multilinear polynomials, `supported_size` is 2 to the number of
    ///   variables.
    ///
    /// `supported_log_size` should be in range `1..=params.log_size`
    fn trim(
        &self,
        supported_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError>;

    /// Build SRS for testing.
    ///
    /// - For univariate polynomials, `supported_size` is the maximum degree.
    /// - For multilinear polynomials, `supported_size` is the number of
    ///   variables.
    ///
    /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
    /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
    fn gen_srs_for_testing<R: Rng>(rng: &mut R, supported_size: usize) -> Result<Self, PCSError>;
}

pub trait NewPolynomialCommitmentScheme<F: PrimeField>: Clone + Debug {
    type Param: Clone + Debug + CanonicalSerialize + CanonicalDeserialize;
    type ProverParam: Clone + Debug + CanonicalSerialize + CanonicalDeserialize;
    type VerifierParam: Clone + Debug + CanonicalSerialize + CanonicalDeserialize;
    type Polynomial: Clone + Debug + Hash + PartialEq + Eq;
    type Point: Clone + Ord + Debug + Sync + Hash + PartialEq + Eq;
    type Commitment: Clone
        + Debug
        + AsRef<[Self::CommitmentChunk]>
        + CanonicalSerialize
        + CanonicalDeserialize;
    type CommitmentChunk: Clone + Debug + Default;

    fn setup(
        poly_size: usize,
        batch_size: usize,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self::Param, PCSError>;

    fn trim(
        param: &Self::Param,
        poly_size: usize,
        batch_size: usize,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), PCSError>;

    fn commit(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, PCSError>;

    fn open(
        pp: &Self::ProverParam,
        poly: &Self::Polynomial,
        comm: &Self::Commitment,
        point: &Self::Point,
        eval: &F,
        // transcript: &mut IOPTranscript<F>,
        transcript: &mut impl TranscriptWrite<Self::CommitmentChunk, F>,
    ) -> Result<(), PCSError>;

    fn verify(
        vp: &Self::VerifierParam,
        // comm: &Self::Commitment,
        point: &Self::Point,
        eval: &F,
        transcript: &mut impl TranscriptRead<Self::CommitmentChunk, F>,
        rng: &mut ChaCha8Rng,
    ) -> Result<(), PCSError>;
}

pub trait BasefoldExtParams: Debug {
    fn get_reps() -> usize;

    fn get_rate() -> usize;

    fn get_basecode_rounds() -> usize;

    fn get_rs_basecode() -> bool;

    fn get_code_type() -> String;
}
