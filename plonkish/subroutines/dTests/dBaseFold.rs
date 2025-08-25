use deNetwork::DeMultiNet;
use rand::rngs::OsRng;
mod common;
use plonkish_backend::{
    poly::multilinear::{MultilinearPolynomial},
    backend::{self, PlonkishBackend, PlonkishCircuit},
    frontend::halo2::{circuit::VanillaPlonk, CircuitExt, Halo2Circuit},
    halo2_curves::{bn256::{Bn256, Fr}, secp256k1::Fp},
    pcs::{PolynomialCommitmentScheme, multilinear::{MultilinearKzg,Basefold,MultilinearBrakedown, ZeromorphFri, BasefoldExtParams }, univariate::Fri},
    util::{
        end_timer, start_timer,
        test::std_rng,
        transcript::{InMemoryTranscript, Blake2sTranscript, Keccak256Transcript, TranscriptRead, TranscriptWrite, Blake2s256Transcript},
	hash::{Keccak256,Blake2s256,Blake2s},
	arithmetic::{PrimeField,sum},
	code::{BrakedownSpec6, BrakedownSpec1},
	new_fields::{Mersenne127, Mersenne61},
	mersenne_61_mont::Mersenne61Mont,
	ff_255::{ff255::Ft255, ft127::Ft127, ft63::Ft63},
	goldilocksMont::GoldilocksMont
    },
};
use ark_std::fs::create_dir_all;
use serde::{Serialize, de::DeserializeOwned};
use subroutines::BasefoldParams::*;
const OUTPUT_DIR: &str = "./bench_data/pcs";
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

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

use crate::common::test_rng;
#[derive(Debug)]
struct P {}

impl BasefoldExtParams for P{

    fn get_rate() -> usize{
	return 2;
    }

    fn get_basecode_rounds() -> usize{
	return 2;
    }

    fn get_reps() -> usize{
	return 1;
    }

    fn get_rs_basecode() -> bool{
	false
    }
    fn get_code_type() -> String{
        "random".to_string()
    }
}
impl System {
    fn all() -> Vec<System> {
        vec![
            System::MultilinearKzg,
	    System::Basefold61Mersenne,
	    System::Basefold256,
	    System::Brakedown,
	    System::BasefoldBlake2s,
	    System::BrakedownBlake2s,
	    System::ZeromorphFri
	    
	
        ]
    }

    fn output_path(&self) -> String {
        format!("{OUTPUT_DIR}/open_{self}")
    }

    fn output(&self) -> File {

        let path = self.output_path();
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

    fn commit_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/commit_{self}")
    }

    fn size_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/size_{self}")
    }

    fn verify_output_path(&self) -> String {
        format!("{OUTPUT_DIR}/verify_{self}")
    }    

    fn commit_output(&self) -> File {

        // OpenOptions::new()
        //     .append(true)
        //     .create(true)
        //     .open(self.commit_output_path())
        //     .unwrap()
        let path = self.commit_output_path();
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


    fn verify_output(&self) -> File {
        // OpenOptions::new()
        //     .append(true)
        //     .create(true)
        //     .open(self.verify_output_path())
        //     .unwrap()
        let path = self.verify_output_path();
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


    fn bench(&self, k: usize) {
	type Kzg = MultilinearKzg<Bn256>;
	type Brakedown = MultilinearBrakedown<Fp, Keccak256, BrakedownSpec6>;
	type Brakedown127 = MultilinearBrakedown<Fp, Blake2s, BrakedownSpec6>;	
	type BrakedownBlake2s = MultilinearBrakedown<GoldilocksMont, Blake2s, BrakedownSpec1>;		
    test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Ten>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s)
    //     match self {
	//     System::MultilinearKzg => test_single_commit::<_, Kzg, Blake2sTranscript<_>>(k, System::MultilinearKzg),
	//     System::Basefold61Mersenne => test_single_commit::<Mersenne127, Basefold<Mersenne127,Blake2s,P>, Blake2sTranscript<_>>(k, System::Basefold61Mersenne),
	//     System::Basefold256 => test_single_commit::<Fr, Basefold<Fr,Blake2s256,BasefoldFri>, Blake2s256Transcript<_>>(k, System::Basefold256),	    
	//     System::Brakedown => test_single_commit::<Fp, Brakedown, Keccak256Transcript<_>>(k,System::Brakedown),
	//     System::BasefoldBlake2s => {
	//     match k {
	//     	  10 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Ten>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  11 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Eleven>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  12 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Twelve>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  13 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Thirteen>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  14 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Fourteen>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  15 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Fifteen>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  16 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Sixteen>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  17 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Seventeen>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  18 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Eighteen>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  19 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Nineteen>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  20 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,Twenty>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  21 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,TwentyOne>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  22 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,TwentyTwo>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  23 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,TwentyThree>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  24 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,TwentyFour>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  25 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,TwentyFive>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),
	// 	  26 => test_single_commit::<GoldilocksMont, Basefold<GoldilocksMont,Blake2s,TwentySix>, Blake2sTranscript<_>>(k,System::BasefoldBlake2s),		  
	// 	  _ => {}
	// 	   }
	//     }
	//     System::BrakedownBlake2s => test_single_commit::<GoldilocksMont, BrakedownBlake2s, Blake2sTranscript<_>>(k,System::BrakedownBlake2s),
	//     System::ZeromorphFri => test_single_commit::<Fr, ZeromorphFri<Fri<Fr,Blake2s>>, Blake2sTranscript<_>>(k, System::ZeromorphFri)
            
	// }
    }
}

impl Display for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            System::Basefold256 => write!(f, "basefold256"),
	    System::Basefold61Mersenne => write!(f, "basefold61Mersenne"),
	    System::MultilinearKzg => write!(f, "kzg"),
	    System::Brakedown => write!(f, "brakedown"),
	    System::BrakedownBlake2s => write!(f,"brakedown_blake"),
	    System::BasefoldBlake2s => write!(f,"basefold_blake"),
	    System::ZeromorphFri => write!(f, "zeromorph_fri")
        }
    }
}

// fn test_single_helper<F: PrimeField>(
//     params: &BasefoldParams<F>,
//     poly: &Arc<DenseMultilinearExtension<F>>,
//     rng: &mut ChaCha8Rng,
// ) -> Result<(), PCSError> {
//     let nv = poly.num_vars;
//     assert_ne!(nv, 0);
//     // let num_party_vars = Net::n_parties().log_2();
//     let (pp, vp) = Basefold::<F, Ten>::trim(&params, 1 << poly.num_vars(), 1).unwrap();
//     let point = if Net::am_master() {
//         let point: Vec<_> = (0..(nv)).map(|_| F::rand(rng)).collect();
//         Net::recv_from_master_uniform(Some(point))
//     } else {
//         Net::recv_from_master_uniform(None)
//     };

//     let eval = poly.evaluate(&point).unwrap();
//     let mut transcript = Blake2s256Transcript::new(());

//     let comm = Basefold::<F, Ten>::commit(&pp, &poly).unwrap();
//     Basefold::<F, Ten>::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();

//     let proof = transcript.into_proof();

//     let mut transcript = Blake2s256Transcript::from_proof((), proof.as_slice());
//     let b = Basefold::<F, Ten>::verify(
//         &vp,
//         // &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
//         &point,
//         &eval,
//         &mut transcript,
//         rng,
//     );
//     // if Net::am_master() {
//     //     let com = com.unwrap();

//     //     let value = d_evaluate_mle(poly, Some(&point)).unwrap();
//     //     assert!(DeMkzg::verify(&vk, &com, &point, &value, &proof)?);

//     //     let value = E::ScalarField::rand(rng);
//     //     assert!(!DeMkzg::verify(&vk, &com, &point, &value, &proof)?);
//     // } else {
//     //     // d_evaluate_mle(poly, None);
//     // }

//     Ok(())
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum System {
    MultilinearKzg,
    Basefold256,
    Basefold61Mersenne,
    BasefoldBlake2s,
    Brakedown,
    BrakedownBlake2s,
    ZeromorphFri
}

fn test_single_commit<F, Pcs, T>(k: usize, pcs : System )
where
   F: PrimeField + Serialize + DeserializeOwned,
   Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
   T: TranscriptRead<Pcs::CommitmentChunk, F>
        + TranscriptWrite<Pcs::CommitmentChunk, F>
        + InMemoryTranscript<Param = ()>,
{
    let mut rng = test_rng();
    let point = if Net::am_master() {
        let point: Vec<_> = (0..(k))
            .map(|_| F::random(rng.clone()))
            .collect();
        Net::recv_from_master_uniform(Some(point))
    } else {
        Net::recv_from_master_uniform(None)
    };
    let timer = start_timer(|| format!("PCS setup and trim -{k}"));
    let mut rng = OsRng;
    let poly_size = 1 << k;
    let param = Pcs::setup(poly_size, 1, &mut rng).unwrap();
    let trim_t = Instant::now();
    let (pp,vp) = Pcs::trim(&param, poly_size, 1).unwrap();



    let timer = start_timer(|| format!("commit -{k}"));
    let mut transcript = T::new(());

    let poly = MultilinearPolynomial::rand(k,OsRng);

    let sample_size = sample_size(k);

    let mut commit_times = Vec::new();
    let mut times = Vec::new();
    for i in 0..sample_size{

	let cstart = Instant::now();
	let comm = Pcs::commit_and_write(&pp, &poly, &mut transcript).unwrap();
	
	commit_times.push(cstart.elapsed());

	let start = Instant::now();
	let point = transcript.squeeze_challenges(k);
	let eval = poly.evaluate(point.as_slice());
	transcript.write_field_element(&eval).unwrap();	
	Pcs::open(&pp, &poly, &comm, &point, &eval, &mut transcript).unwrap();
	times.push(start.elapsed());
    }
    let sum = times.iter().sum::<Duration>();
    let csum = commit_times.iter().sum::<Duration>();
    
    let avg = sum / sample_size as u32;
    let cavg = csum /sample_size as u32;

    
    writeln!(&mut pcs.commit_output(), "{k}, {}", cavg.as_millis()).unwrap();
    writeln!(&mut pcs.output(), "{k}, {}", avg.as_millis()).unwrap();    
    
    let proof = transcript.into_proof();




    let timer = start_timer(|| format!("verify-{k}"));
    let result = {
	let mut transcript = T::from_proof((),proof.as_slice());
	let mut start_size = 0;
	while(transcript.read_commitment().is_ok()){
	    start_size = start_size + 1;
	}

	let mut transcript = T::from_proof((),proof.as_slice());
	let now = Instant::now();
	let b = Pcs::verify(
            &vp,
            &Pcs::read_commitment(&vp, &mut transcript).unwrap(),
            &transcript.squeeze_challenges(k),
            &transcript.read_field_element().unwrap(),
            &mut transcript
	);
	writeln!(&mut pcs.verify_output(), "{:?}: {:?}", k, now.elapsed().as_millis()).unwrap();
	let mut end_size = 0;
	while(transcript.read_commitment().is_ok()){
	    end_size = end_size + 1;
	}
	writeln!(&mut pcs.size_output(), "{:?} {:?} : {:?}", pcs, k, (start_size - end_size)*256);
	b
    };

    end_timer(timer);
    assert_eq!(result,Ok(()));

}

fn main() {
    common::network_run(|| {
        let (systems, k_range) = ([System::Basefold256], Range { start: 11, end: 12 });
    
        k_range.for_each(|k| systems.iter().for_each(|system| system.bench(k)));
    });
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
