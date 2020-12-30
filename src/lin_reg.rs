extern crate serde;
#[macro_use()]
extern crate serde_derive;
use std::io::{ prelude::*, BufReader };
use std::{ path::Path, fs::File, vec::Vec, error::Error };
use rand::{ thread_rng, seq::SliceRandom };
