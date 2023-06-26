use std::fs::File;
use std::io::Read;

use ndarray::prelude::*;
use crate::Error;
use crate::utils::save_load::{save_model_params, load_model_params};

use serde::{Serialize, Deserialize};
use serde_json;

pub trait ActivationTrait {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, dvalues: Array2<f64>) -> Array2<f64>;
} 

pub trait NeuralNetworkTrait<T>: Serialize {
    type Model: 'static + Deserialize<'static>;

    fn new (input_size: i32, output_size: i32) -> Self;
    fn forward(&mut self, inputs: T) -> T;

    fn save(&mut self, path: String) -> Result<(), Error> {
        save_model_params(path, self.params())
    }

    fn load(&mut self, path: String) -> Result<(), Error>
    where
        Self: Serialize + Deserialize<'static>,
    {
        let loaded_params = load_model_params::<Self>(path)?;
        *self = loaded_params;

        Ok(())
    }

    fn params(&mut self) -> String {
        let serialized = serde_json::to_string(&self).unwrap();
        serialized
    }
}