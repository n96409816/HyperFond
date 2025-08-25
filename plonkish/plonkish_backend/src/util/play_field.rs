use crate::util::{
    arithmetic::{modulus, Field},
    BigUint,
};
use core::fmt;
use core::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use ff::{BatchInvert, PrimeFieldBits};
use halo2_curves::ff::PrimeField;
use rand::RngCore;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display, Formatter};
use std::ops::{BitAnd, Shr};
use std::time::Instant;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[derive(PrimeField, Serialize, Deserialize, Hash)]
#[PrimeFieldModulus = "17"]
#[PrimeFieldGenerator = "3"]
#[PrimeFieldReprEndianness = "little"]
pub struct PlayField([u64; 1]);
