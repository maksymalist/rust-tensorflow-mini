use crate::utils::layer_dense::LayerDense;
use ndarray::prelude::*;
pub struct OptimizerSGD {
    pub learning_rate: f64,
}

impl OptimizerSGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn update_params(&self, layer: &mut LayerDense, epoch: usize) {
        let weight_adj = layer.dweights.mapv(|x| x * -self.learning_rate);
        let bias_adj = layer.dbiases.mapv(|x| x * -self.learning_rate);

        layer.weights = layer.weights.clone() + weight_adj.t();
        layer.biases = layer.biases.clone() + bias_adj.t();
    }
}
