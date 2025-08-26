# A Transparent and Post-Quantum Distributed SNARK with Polylogarithmic Communication
This is the artifact for the paper HyperFond: A Transparent and Post-Quantum Distributed SNARK with Polylogarithmic Communication

## Acknowledgment

We would like to acknowledge that our implementation is based on the base fold repository (https://github.com/hadasz/plonkish_basefold) and hyperplonk repository (https://github.com/EspressoSystems/hyperplonk). Specifically, we forked these repositories and built upon them for our development work. 

## Installation
We recommend using Ubuntu for running the experiments.
1. Install Rust nightly:
   - `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   - `rustup override set nightly` 1.89.0-nightly
2. Compile the experiments:
   - `git clone git@github.com:MengLing-L/_HyperFond.git`
   - `cd _HyperFond/plonkish`
   - `python3 compile.py`

## Testing on Localhost

### SNARK Baseline: HyperPlonk + Basefold
- `target/release/examples/basefold_proof_system `
  
Note: The default number of variables is `22`. To change this, modify `system.bench(22, circuit);` in `plonkish/benchmark/benches/basefold_proof_system.rs`.

### Distributed SNARK: HyperFond 
1. By default, the number of parties is set to `2` (configuration files in [plonkish/2](plonkish/2)).
2. In terminal 1:
   `target/release/examples/distributed_basefold_proof_system  0 2 22`
3. In terminal 2:
    `target/release/examples/distributed_basefold_proof_system  1 2 22`

Note: 
- Change `22` to your desired number of variables.
- To use `4` parties instead of `2`:
  - Use configuration files from [plonkish/4](plonkish/4)
  - Run for each party (0-3)
  `target/release/examples/distributed_basefold_proof_system {party_number} 4 22`  
## Remark
This implementation is a preliminary prototype to evaluate our solution's performance and is not yet ready for productive use.
