use ndarray::prelude::*;

use super::ActivationTrait;

pub struct ActivationReLU {
    pub output: Array2<f64>
}

impl ActivationTrait for ActivationReLU {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        self.output = inputs.mapv(|x| x.max(0.0));
        self.output.clone()
    }
}

impl ActivationReLU {
    pub fn new() -> Self {
        Self {
            output: Array2::zeros((1, 1))
        }
    }
}