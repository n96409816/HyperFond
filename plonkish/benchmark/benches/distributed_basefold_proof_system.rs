use ark_std::fs::create_dir_all;
use ark_std::path::PathBuf;
use benchmark::{
    espresso,
    halo2::{AggregationCircuit, Sha256Circuit},
    BasefoldParams::*,
    Math,
};
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet, Stats};
use espresso_hyperplonk::{prelude::MockCircuit, HyperPlonkSNARK};
use espresso_subroutines::{MultilinearKzgPCS, PolyIOP, PolynomialCommitmentScheme};
use ff::Field;
use halo2_proofs::{
    circuit,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof},
    poly::kzg::{
        commitment::ParamsKZG,
        multiopen::{ProverGWC, VerifierGWC},
        strategy::SingleStrategy,
    },
    transcript::{Blake2bRead, Blake2bWrite, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use itertools::Itertools;
use plonkish_backend::{
    backend::{
        self,
        hyperplonk::{
            util::{
                rand_vanilla_plonk_assignment, rand_vanilla_plonk_with_lookup_assignment,
                vanilla_plonk_expression,
            },
            HyperPlonkProverParam, HyperPlonkVerifierParam,
        },
        PlonkishBackend, PlonkishCircuit,
    },
    frontend::halo2::{circuit::VanillaPlonk, CircuitExt, Halo2Circuit},
    halo2_curves::{
        bn256::{Bn256, Fr},
        secp256k1::Fp,
    },
    pcs::multilinear,
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        evaluate, SumCheck, VirtualPolynomial,
    },
    poly::multilinear::rotation_eval,
    util::{
        end_timer,
        goldilocksMont::GoldilocksMont,
        hash::{Blake2s, Blake2s256},
        mersenne_61_mont::Mersenne61Mont,
        new_fields::{Mersenne127, Mersenne61},
        start_timer,
        test::{rand_vec, std_rng},
        transcript::{Blake2sTranscript, InMemoryTranscript, Keccak256Transcript},
    },
};
use structopt::StructOpt;

use std::{
    env::args,
    fmt::Display,
    fs::{create_dir, File, OpenOptions},
    io::Write,
    iter,
    ops::Range,
    path::Path,
    time::{Duration, Instant},
};
#[derive(Debug)]
struct P {}

impl plonkish_backend::pcs::multilinear::BasefoldExtParams for P {
    fn get_rate() -> usize {
        return 2;
    }

    fn get_basecode_rounds() -> usize {
        return 2;
    }

    fn get_rs_basecode() -> bool {
        false
    }

    fn get_reps() -> usize {
        return 1000;
    }
    fn get_code_type() -> String {
        "random".to_string()
    }
}
const OUTPUT_DIR: &str = "./bench_data/";

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(long)]
    dory: bool,

    #[structopt(long)]
    jellyfish: bool,

    num_vars: usize,
}

fn main() {
    let opt = Opt::from_args();
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);

    let circuit = Circuit::VanillaPlonk;

    let system = System::HyperPlonk;

    system.bench(opt.num_vars - Net::n_parties().log_2(), circuit)
}


fn bench_hyperplonk_18_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, Eighteen8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

fn bench_hyperplonk_19_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, Nineteen8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

   
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

fn bench_hyperplonk_20_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, Twenty8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

    
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

fn bench_hyperplonk_21_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, TwentyOne8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

    
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

fn bench_hyperplonk_22_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, TwentyTwo8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

    
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

fn bench_hyperplonk_23_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, TwentyThree8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

    
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

fn bench_hyperplonk_24_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, TwentyFour8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

    
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

fn bench_hyperplonk_25_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
    type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, TwentyFive8>;
    type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

    
    let circuit = C::rand(k, std_rng());
    let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
    let circuit_info = circuit.circuit_info().unwrap();
    let instances = circuit.instances();

    let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
    let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
    end_timer(timer);

    let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
    HyperPlonk::d_preprocess_p(&param, &circuit_info).unwrap();
    HyperPlonk::d_preprocess_v(&param, &circuit_info).unwrap();
    end_timer(timer);

    let sub_prover_setup_filepath = format!("../data/SubProver{}-{}.paras", Net::party_id(), k);
    let pp = HyperPlonkProverParam::read_from_file(&sub_prover_setup_filepath).unwrap();

    drop(param);
    drop(circuit_info);

    let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
    let start = Instant::now();
    let mut transcript = Blake2sTranscript::default();
    HyperPlonk::d_prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
    let proof = transcript.into_proof();
    let proofs = Net::send_to_master(&proof);
    println!(
        "proving for {} variables: {:?}",
        k + Net::n_parties().log_2(),
        start.elapsed(),
    );

    end_timer(_timer);
    drop(pp);
    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

    if Net::am_master() {
        let verifier_setup_filepath = format!("../data/Verifier{}-{}.paras", Net::party_id(), k);
        let vp = HyperPlonkVerifierParam::read_from_file(&verifier_setup_filepath).unwrap();
        let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
        let start = Instant::now();
        println!("proof length {}.", proofs.clone().unwrap().len(),);
        let mut i = 0;
        for proof in proofs.unwrap().iter() {
            if i == 0 {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::d_verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            } else {
                let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
                let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();
                assert!(accept);
                println!("accept :{:?}.", accept);
            }
            i = i + 1;
        }
        println!(
            "verifiy for {} variables: {:?}",
            k + Net::n_parties().log_2(),
            start.elapsed(),
        );
        end_timer(_timer);
    }

    Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
}

// fn bench_hyperplonk_26_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
//     type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, TwentySix8>;
//     type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

//     let m = if Net::n_parties().log_2() == 1 {
//         Net::n_parties().log_2() + 1
//     } else {
//         Net::n_parties().log_2()
//     };

//     let expression = vanilla_plonk_expression(m);
//     let degree = expression.degree();
//     let (pp_sum, vp_sum) = ((), ());
//     let (polys, challenges) = rand_vanilla_plonk_assignment(m, std_rng(), std_rng());
//     let (polys, challenges, y) = (polys, challenges, rand_vec(m, std_rng()));

//     let ys = [y];

//     let circuit = C::rand(k, std_rng());
//     let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
//     let circuit_info = circuit.circuit_info().unwrap();
//     let instances = circuit.instances();

//     let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
//     let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
//     end_timer(timer);

//     let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
//     let (pp, vp) = HyperPlonk::preprocess(&param, &circuit_info).unwrap();
//     end_timer(timer);

//     // let proof = sample(System::HyperPlonk, String::from("Fp - ecdsa"), k, || {
//     //     let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
//     //     let mut transcript = Blake2sTranscript::default();
//     //     HyperPlonk::prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
//     //     let proofs = transcript.into_proof();
//     //     proofs
//     // });

//     let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
//     let mut proof_sum = Blake2sTranscript::default().into_proof();
//     let start = Instant::now();
//     if Net::am_master() {
//         proof_sum = {
//             let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &ys);
//             let mut transcript = Blake2sTranscript::default();
//             ClassicSumCheck::<EvaluationsProver<_>>::prove(
//                 &pp_sum,
//                 m,
//                 virtual_poly,
//                 GoldilocksMont::ZERO,
//                 &mut transcript,
//             )
//             .unwrap();
//             transcript.into_proof()
//         };
//     }
//     let mut transcript = Blake2sTranscript::default();
//     HyperPlonk::prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
//     let proof = transcript.into_proof();
//     let proofs = Net::send_to_master(&proof);
//     println!(
//         "proving for {} variables: {:?}",
//         k + Net::n_parties().log_2(),
//         start.elapsed(),
//     );

//     end_timer(_timer);

//     Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

//     if Net::am_master() {
//         // } else {
//         //     end_timer!(commit_timer);
//         //     Ok((None, ()))

//         let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
//         let start = Instant::now();
//         // let accept = verifier_sample(System::HyperPlonk, String::from("fp_vanilla"), k, || {
//         //     let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
//         //     HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok()
//         // });
//         println!("proof length {}.", proofs.clone().unwrap().len(),);
//         for proof in proofs.unwrap().iter() {
//             let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
//             let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();

//             let mut t1 = Blake2sTranscript::from_proof((), proof.as_slice());
//             let end_size = t1.into_proof().len();

//             writeln!(
//                 &mut (System::HyperPlonk).size_output(),
//                 "{k} : {:?}",
//                 (end_size) * 8
//             )
//             .unwrap();

//             assert!(accept);
//             println!("accept :{}.", accept,);
//         }

//         let accept = {
//             let mut transcript = Blake2sTranscript::from_proof((), proof_sum.as_slice());
//             let (x_eval, x) = ClassicSumCheck::<EvaluationsProver<_>>::verify(
//                 &vp_sum,
//                 m,
//                 degree,
//                 GoldilocksMont::ZERO,
//                 &mut transcript,
//             )
//             .unwrap();
//             let evals = expression
//                 .used_query()
//                 .into_iter()
//                 .map(|query| {
//                     let evaluate_for_rotation =
//                         polys[query.poly()].evaluate_for_rotation(&x, query.rotation());
//                     let eval = rotation_eval(&x, query.rotation(), &evaluate_for_rotation);
//                     (query, eval)
//                 })
//                 .collect();
//             x_eval == evaluate(&expression, m, &evals, &challenges, &[&ys[0]], &x)
//         };
//         assert!(accept);
//         println!(
//             "verifiy for {} variables: {:?}",
//             k + Net::n_parties().log_2(),
//             start.elapsed(),
//         );
//         end_timer(_timer);
//     }

//     Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
// }

// fn bench_hyperplonk_27_8<C: CircuitExt<GoldilocksMont>>(k: usize) {
//     type Basefold = multilinear::Basefold<GoldilocksMont, Blake2s256, TwentySeven8>;
//     type HyperPlonk = backend::hyperplonk::HyperPlonk<Basefold>;

//     let m = if Net::n_parties().log_2() == 1 {
//         Net::n_parties().log_2() + 1
//     } else {
//         Net::n_parties().log_2()
//     };

//     let expression = vanilla_plonk_expression(m);
//     let degree = expression.degree();
//     let (pp_sum, vp_sum) = ((), ());
//     let (polys, challenges) = rand_vanilla_plonk_assignment(m, std_rng(), std_rng());
//     let (polys, challenges, y) = (polys, challenges, rand_vec(m, std_rng()));

//     let ys = [y];

//     let circuit = C::rand(k, std_rng());
//     let circuit = Halo2Circuit::new::<HyperPlonk>(k, circuit);
//     let circuit_info = circuit.circuit_info().unwrap();
//     let instances = circuit.instances();

//     let timer = start_timer(|| format!("hyperplonk_setup-{k}"));
//     let param = HyperPlonk::setup(&circuit_info, std_rng()).unwrap();
//     end_timer(timer);

//     let timer = start_timer(|| format!("hyperplonk_preprocess-{k}"));
//     let (pp, vp) = HyperPlonk::preprocess(&param, &circuit_info).unwrap();
//     end_timer(timer);

//     // let proof = sample(System::HyperPlonk, String::from("Fp - ecdsa"), k, || {
//     //     let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
//     //     let mut transcript = Blake2sTranscript::default();
//     //     HyperPlonk::prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
//     //     let proofs = transcript.into_proof();
//     //     proofs
//     // });

//     let _timer = start_timer(|| format!("hyperplonk_prove-{k}"));
//     let mut proof_sum = Blake2sTranscript::default().into_proof();
//     let start = Instant::now();
//     if Net::am_master() {
//         proof_sum = {
//             let virtual_poly = VirtualPolynomial::new(&expression, &polys, &challenges, &ys);
//             let mut transcript = Blake2sTranscript::default();
//             ClassicSumCheck::<EvaluationsProver<_>>::prove(
//                 &pp_sum,
//                 m,
//                 virtual_poly,
//                 GoldilocksMont::ZERO,
//                 &mut transcript,
//             )
//             .unwrap();
//             transcript.into_proof()
//         };
//     }
//     let mut transcript = Blake2sTranscript::default();
//     HyperPlonk::prove(&pp, &circuit, &mut transcript, std_rng()).unwrap();
//     let proof = transcript.into_proof();
//     let proofs = Net::send_to_master(&proof);
//     println!(
//         "proving for {} variables: {:?}",
//         k + Net::n_parties().log_2(),
//         start.elapsed(),
//     );

//     end_timer(_timer);

//     Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });

//     if Net::am_master() {
//         // } else {
//         //     end_timer!(commit_timer);
//         //     Ok((None, ()))

//         let _timer = start_timer(|| format!("hyperplonk_verify-{k}"));
//         let start = Instant::now();
//         // let accept = verifier_sample(System::HyperPlonk, String::from("fp_vanilla"), k, || {
//         //     let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
//         //     HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok()
//         // });
//         println!("proof length {}.", proofs.clone().unwrap().len(),);
//         for proof in proofs.unwrap().iter() {
//             let mut transcript = Blake2sTranscript::from_proof((), proof.as_slice());
//             let accept = HyperPlonk::verify(&vp, instances, &mut transcript, std_rng()).is_ok();

//             let mut t1 = Blake2sTranscript::from_proof((), proof.as_slice());
//             let end_size = t1.into_proof().len();

//             writeln!(
//                 &mut (System::HyperPlonk).size_output(),
//                 "{k} : {:?}",
//                 (end_size) * 8
//             )
//             .unwrap();

//             assert!(accept);
//             println!("accept :{}.", accept,);
//         }

//         let accept = {
//             let mut transcript = Blake2sTranscript::from_proof((), proof_sum.as_slice());
//             let (x_eval, x) = ClassicSumCheck::<EvaluationsProver<_>>::verify(
//                 &vp_sum,
//                 m,
//                 degree,
//                 GoldilocksMont::ZERO,
//                 &mut transcript,
//             )
//             .unwrap();
//             let evals = expression
//                 .used_query()
//                 .into_iter()
//                 .map(|query| {
//                     let evaluate_for_rotation =
//                         polys[query.poly()].evaluate_for_rotation(&x, query.rotation());
//                     let eval = rotation_eval(&x, query.rotation(), &evaluate_for_rotation);
//                     (query, eval)
//                 })
//                 .collect();
//             x_eval == evaluate(&expression, m, &evals, &challenges, &[&ys[0]], &x)
//         };
//         assert!(accept);
//         println!(
//             "verifiy for {} variables: {:?}",
//             k + Net::n_parties().log_2(),
//             start.elapsed(),
//         );
//         end_timer(_timer);
//     }

//     Net::recv_from_master_uniform(if Net::am_master() { Some(1usize) } else { None });
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum System {
    HyperPlonk,
}

impl System {
    fn all() -> Vec<System> {
        vec![System::HyperPlonk]
    }

    fn output_path(&self) -> String {
        format!("{OUTPUT_DIR}/hyperplonk-basefold")
    }

    fn output(&self) -> File {
        OpenOptions::new()
            .append(true)
            .open(self.output_path())
            .unwrap()
    }

    fn verifier_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/{self}-kzg-verifier")
    }

    fn size_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/{self}-kzg-size")
    }

    fn size_output(&self) -> File {
        let path = self.size_output_path();
        let test_path = Path::new(&path);
        if let Some(parent) = test_path.parent() {
            create_dir_all(parent).unwrap();
        }
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(path.clone())
            .unwrap_or_else(|e| panic!("can not open {}: {}", path.clone(), e))
    }
    fn verifier_output(&self) -> File {
        let path = self.verifier_output_path();
        let test_path = Path::new(&path);
        if let Some(parent) = test_path.parent() {
            create_dir_all(parent).unwrap();
        }
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(path.clone())
            .unwrap_or_else(|e| panic!("can not open {}: {}", path.clone(), e))
    }

    fn support(&self, circuit: Circuit) -> bool {
        match self {
            System::HyperPlonk => match circuit {
                Circuit::VanillaPlonk | Circuit::ECDSA | Circuit::Sha256 => true,
            },
        }
    }

    fn bench(&self, k: usize, circuit: Circuit) {
        if !self.support(circuit) {
            println!("skip benchmark on {circuit} with {self} because it's not compatible");
            return;
        }

        println!("start benchmark on 2^{k} {circuit} with {self}");

        // match self {
        //     System::HyperPlonk => bench_hyperplonk_256::<VanillaPlonk<Fr>>(k)

        match k {
            18 => bench_hyperplonk_18_8::<VanillaPlonk<GoldilocksMont>>(k),
            19 => bench_hyperplonk_19_8::<VanillaPlonk<GoldilocksMont>>(k),
            20 => bench_hyperplonk_20_8::<VanillaPlonk<GoldilocksMont>>(k),
            21 => bench_hyperplonk_21_8::<VanillaPlonk<GoldilocksMont>>(k),
            22 => bench_hyperplonk_22_8::<VanillaPlonk<GoldilocksMont>>(k),
            23 => bench_hyperplonk_23_8::<VanillaPlonk<GoldilocksMont>>(k),
            24 => bench_hyperplonk_24_8::<VanillaPlonk<GoldilocksMont>>(k),
            25 => bench_hyperplonk_25_8::<VanillaPlonk<GoldilocksMont>>(k),
            // 26 => bench_hyperplonk_26_8::<VanillaPlonk<GoldilocksMont>>(k),
            _ => {}
        }

        // }
    }
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            System::HyperPlonk => write!(f, "hyperplonk"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Circuit {
    VanillaPlonk,
    ECDSA,
    Sha256,
}

impl Circuit {
    fn min_k(&self) -> usize {
        match self {
            Circuit::VanillaPlonk => 4,
            Circuit::ECDSA => 4,
            Circuit::Sha256 => 17,
        }
    }
}

impl Display for Circuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Circuit::VanillaPlonk => write!(f, "vanilla_plonk"),
            Circuit::ECDSA => write!(f, "ecdsa - fp"),
            Circuit::Sha256 => write!(f, "sha256"),
        }
    }
}

fn parse_args() -> (Vec<System>, Circuit, Range<usize>) {
    let (systems, circuit, k_range) = args().chain(Some("".to_string())).tuple_windows().fold(
        (Vec::new(), Circuit::ECDSA, 10..11),
        |(mut systems, mut circuit, mut k_range), (key, value)| {
            match key.as_str() {
                "--system" => match value.as_str() {
                    "all" => systems = System::all(),
                    "hyperplonk" => systems.push(System::HyperPlonk),
                    _ => panic!(
                        "system should be one of {{all,hyperplonk,halo2,espresso_hyperplonk}}"
                    ),
                },
                "--circuit" => match value.as_str() {
                    "vanilla_plonk" => circuit = Circuit::VanillaPlonk,
                    "aggregation" => circuit = Circuit::ECDSA,
                    "sha256" => circuit = Circuit::Sha256,
                    _ => panic!("circuit should be one of {{aggregation,vanilla_plonk}}"),
                },
                "--k" => {
                    if let Some((start, end)) = value.split_once("..") {
                        k_range = start.parse().expect("k range start to be usize")
                            ..end.parse().expect("k range end to be usize");
                    } else {
                        k_range.start = value.parse().expect("k to be usize");
                        k_range.end = k_range.start + 1;
                    }
                }
                _ => {}
            }
            (systems, circuit, k_range)
        },
    );
    if k_range.start < circuit.min_k() {
        panic!("k should be at least {} for {circuit:?}", circuit.min_k());
    }
    let mut systems = systems.into_iter().sorted().dedup().collect_vec();
    if systems.is_empty() {
        systems = System::all();
    };
    (systems, circuit, k_range)
}

fn create_output(systems: &[System]) {
    if !Path::new(OUTPUT_DIR).exists() {
        create_dir(OUTPUT_DIR).unwrap();
    }
    for system in systems {
        File::create(system.output_path()).unwrap();
        File::create(system.verifier_output_path()).unwrap();
        File::create(system.size_output_path()).unwrap();
    }
}

fn sample<T>(system: System, key: String, k: usize, prove: impl Fn() -> T) -> T {
    let mut proof = None;
    let sample_size = sample_size(k);
    let sum = iter::repeat_with(|| {
        let start = Instant::now();
        proof = Some(prove());
        start.elapsed()
    })
    .take(sample_size)
    .sum::<Duration>();
    let avg = sum / sample_size as u32;
    writeln!(&mut system.output(), "{k} : {}", avg.as_millis()).unwrap();
    println!("{}", avg.as_millis());
    proof.unwrap()
}

fn verifier_sample<T>(system: System, key: String, k: usize, prove: impl Fn() -> T) -> T {
    let mut proof = None;
    let sample_size = sample_size(k);
    let sum = iter::repeat_with(|| {
        let start = Instant::now();
        proof = Some(prove());
        start.elapsed()
    })
    .take(sample_size)
    .sum::<Duration>();
    let avg = sum / sample_size as u32;
    writeln!(&mut system.verifier_output(), "{k} : {}", avg.as_millis()).unwrap();
    proof.unwrap()
}

fn sample_size(k: usize) -> usize {
    if k < 16 {
        20
    } else if k < 20 {
        5
    } else {
        1
    }
}
