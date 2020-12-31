extern crate serde;
#[macro_use()]
extern crate serde_derive;
use std::io::{ prelude::*, BufReader };
use std::{ path::Path, fs::File, vec::Vec, error::Error };
use rand::{ thread_rng, seq::SliceRandom };

use ml_utils::datasets::get_boston_records_from_file;

pub fn run() -> Result<(), Box<dyn Error>> {
    let fl = "data/housing.csv";
    let mut data = get_boston_records_from_file(&fl);

    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size: f64 = test_size.round() as usize;
    let (test_data, tain_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the predictors and the targets
    let boston_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();
    let boston_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();
}