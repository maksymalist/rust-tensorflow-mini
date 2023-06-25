use ndarray::prelude::*;
use crate::Error;


pub trait ActivationTrait {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64>;
    fn backward(&mut self, dvalues: Array2<f64>) -> Array2<f64>;
} 

pub trait NeuralNetworkTrait<T> {
    fn new (input_size: i32, output_size: i32) -> Self;
    fn forward(&mut self, inputs: T) -> T;
    fn save(&mut self, path: String) -> Result<(), Error> {
        Ok(())
    }
    fn load(&mut self, path: &str) -> Result<(), Error> {
        Ok(())
    }
}