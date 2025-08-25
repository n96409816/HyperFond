
use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};
use rand::{rngs::StdRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{ops::FnOnce, path::PathBuf, sync::Arc};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
struct Opt {
    /// Id
    id: usize,

    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

pub(super) fn network_run<F>(func: F)
where
    F: FnOnce() -> (),
{
    let opt = Opt::from_args();
    Net::init_from_file(opt.input.to_str().unwrap(), opt.id);

    func();

    Net::deinit();
}

pub(super) fn test_rng() -> StdRng {
    let mut seed = [0u8; 32];
    seed[0] = Net::party_id() as u8;
    rand::rngs::StdRng::from_seed(seed)
}

pub(super) fn test_chacharng() -> ChaCha8Rng {
    let mut seed = [0u8; 32];
    seed[0] = Net::party_id() as u8;
    ChaCha8Rng::from_seed(seed)
}
