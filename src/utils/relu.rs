use ndarray::prelude::*;

pub struct ActivationReLU {
    pub output: Array2<f32>
}

impl ActivationReLU {
    pub fn new() -> Self {
        Self {
            output: Array2::zeros((1, 1))
        }
    }
    pub fn forward(&mut self, inputs: Array2<f32>) {
        self.output = inputs.mapv(|x| x.max(0.0));
    }
}