use ndarray::prelude::*;
use serde::{Serialize, Deserialize};

use super::neural_network::ActivationTrait;

#[derive(Serialize, Deserialize)]
pub struct ActivationSoftmax {
    pub output: Array2<f64>,
    pub dinputs: Array2<f64>
}

impl ActivationTrait for ActivationSoftmax {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        let exp_values = inputs.mapv(f64::exp);
        let sum_values = exp_values.sum_axis(Axis(1)).insert_axis(Axis(1));
        let probabilities = exp_values / sum_values;
        self.output = probabilities.clone();
        probabilities
    }

    fn backward(&mut self, dvalues: Array2<f64>) -> Array2<f64> {
        let mut dinputs = Array2::zeros(dvalues.dim());
        for (i, (single_output, single_dvalues)) in self.output.outer_iter().zip(dvalues.outer_iter()).enumerate() {
            let mut jacobian_matrix = Array2::zeros((single_output.len(), single_output.len()));
            for (k, (mut j, &output)) in jacobian_matrix.outer_iter_mut().zip(single_output.iter()).enumerate() {
                j[k] = output - (output * output);
            }
            let mut dvalues_temp = Array2::zeros((1, single_dvalues.len()));
            dvalues_temp.assign(&single_dvalues);
            let temp1 = dvalues_temp.dot(&jacobian_matrix);
            let temp2 = temp1.dot(&single_output.t());
            dinputs.slice_mut(s![i, ..]).assign(&temp2);
        }
        self.dinputs = dinputs.clone();
        dinputs
    }
}

impl ActivationSoftmax {
    pub fn new() -> Self {
        Self {
            output: Array2::zeros((1, 1)),
            dinputs: Array2::zeros((1, 1))
        }
    }
}