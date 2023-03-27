use ndarray::prelude::*;
use std::f64::consts::E;

use super::ActivationTrait;

pub struct ActivationSoftmax {
    pub output: Array2<f64>
}

impl ActivationTrait for ActivationSoftmax {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        let exp_values = inputs.mapv(f64::exp);
        println!("\n exp_values: {:?}", exp_values);
        let sum_values = exp_values.sum_axis(Axis(1)).insert_axis(Axis(1));
        println!("\n\n\n sum_values: {:?} \n\n\n", sum_values);
        let probabilities = exp_values / sum_values;
        println!("\n\n\n probabilities: {:?} \n\n\n", probabilities);
        self.output = probabilities.clone();
        probabilities
    }
}

impl ActivationSoftmax {
    pub fn new() -> Self {
        Self {
            output: Array2::zeros((1, 1))
        }
    }
}