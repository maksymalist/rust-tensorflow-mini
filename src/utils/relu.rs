use ndarray::prelude::*;

use super::ActivationTrait;

pub struct ActivationReLU {
    pub inputs: Array2<f64>,
    pub output: Array2<f64>,
    pub dinputs: Array2<f64>
}

impl ActivationTrait for ActivationReLU {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        self.output = inputs.mapv(|x| x.max(0.0));
        self.inputs = inputs;
        self.output.clone()
    }
    fn backward(&mut self, dvalues: Array2<f64>) -> Array2<f64> {
        self.dinputs = dvalues.mapv(|x| x.max(0.0));
        self.dinputs.clone()
    }
}

impl ActivationReLU {
    pub fn new() -> Self {
        Self {
            inputs: Array2::zeros((1, 1)),
            output: Array2::zeros((1, 1)),
            dinputs: Array2::zeros((1, 1))
        }
    }
}