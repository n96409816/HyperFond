mod brakedown;
mod raa;
// pub mod binary_rs;
pub use brakedown::{
    Brakedown, BrakedownSpec, BrakedownSpec1, BrakedownSpec2, BrakedownSpec3, BrakedownSpec4,
    BrakedownSpec5, BrakedownSpec6,
};
pub use raa::{
    encode_bits, encode_bits_long, encode_bits_ser, repetition_code_long, serial_accumulator_long,
    Permutation,
};

pub trait LinearCodes<F>: Sync + Send {
    fn row_len(&self) -> usize;

    fn codeword_len(&self) -> usize;

    fn num_column_opening(&self) -> usize;

    fn num_proximity_testing(&self) -> usize;

    fn encode(&self, input: impl AsMut<[F]>);
}
