use ndarray::prelude::*;
use crate::utils::layer_dense::LayerDense;
pub struct OptimizerSDG {
    pub learning_rate: f64,
}

impl OptimizerSDG {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    pub fn update_params(&self, layer: &mut LayerDense) {

        // println!(" weights shape: {:?}, {:?}", layer.dweights.shape(), layer.weights.shape());
        // println!(" biases shape: {:?}, {:?}", layer.dbiases.shape(), layer.biases.shape());

        let new_weights  = layer.weights.clone() + layer.dweights.mapv(|x| x * -self.learning_rate).t();
        let new_biases = layer.biases.clone() + layer.dbiases.mapv(|x| x * -self.learning_rate);

        println!(" new weights shape: {:?}, {:?}", new_weights, layer.weights);

        // function that checks if one of the elements in the array is NaN
        for i in new_weights.iter() {
            if i.is_nan() {
                panic!("NaN in weights");
            }
        }

        layer.weights = new_weights;
        layer.biases = new_biases;
    }
}