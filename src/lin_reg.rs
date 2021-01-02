extern crate serde;
#[macro_use()]
use std::io::{ prelude::*, BufReader };
use std::{ path::Path, fs::File, vec::Vec, error::Error, process::exit, env::args };


use rusty_machine;
use rusty_machine::linalg::{ Matrix, Vector };
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::learning::gp::{ GaussianProcess, ConstMean };
use rusty_machine::learning::toolkit::kernel;
use rusty_machine::learning::glm::{ GenLinearModel, Normal };
use rusty_machine::learning::SupModel;
use rusty_machine::analysis::score::neg_mean_squared_error;

extern crate serde_derive;
use rand;
use rand::{ thread_rng, seq::SliceRandom };

use ml_utils::datasets::get_boston_records_from_file;
use ml_utils::sup_metrics::r_squared_score;

pub fn run() -> Result<(), Box<dyn Error>> {
    let fl = "data/housing.csv";
    let mut data = get_boston_records_from_file(&fl);

    data.shuffle(&mut thread_rng());

    // separate out to train and test datasets
    let test_size: f64 = 0.2;
    let test_size: f64 = data.len() as f64 * test_size;
    let test_size = test_size.round() as usize;
    let (test_data, train_data) = data.split_at(test_size);
    let train_size = train_data.len();
    let test_size = test_data.len();

    // differentiate the predictors and the targets
    let boston_x_train: Vec<f64> = train_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_train: Vec<f64> = train_data.iter().map(|r| r.into_targets()).collect();

    let boston_x_test: Vec<f64> = test_data.iter().flat_map(|r| r.into_feature_vector()).collect();
    let boston_y_test: Vec<f64> = test_data.iter().map(|r| r.into_targets()).collect();

    // convert the data into matrices for rusty machine
    let boston_x_train = Matrix::new(train_size, 13, boston_x_train);
    let boston_y_train = Vector::new(boston_y_train);

    let boston_x_test = Matrix::new(test_size, 13, boston_x_test);
    let boston_y_test = Matrix::new(test_size, 1, boston_y_test);

    // create a linear regression model
    let mut lin_model = LinRegressor::default();
    println!("{:?}", lin_model);

    // Train the model
    lin_model.train(&boston_x_train, &boston_y_train);

    // Now will predict
    let predictions = lin_model.predict(&boston_x_test).unwrap();
    let predictions = Matrix::new(test_size, 1, predictions);
    let acc = neg_mean_squared_error(&predictions, &boston_y_test);

    println!("linear regression error: {:?}", acc);
    println!("linear regression R2 score: {:?}", r_squared_score( &boston_y_test.data(), &predictions.data() ));

    Ok(())
}