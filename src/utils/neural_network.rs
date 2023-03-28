use ndarray::prelude::*;
use crate::{ActivationReLU, LayerDense, ActivationSoftmax, Loss};

pub struct  NeuralNetwork {
    layers: Vec<(LayerDense, Activation)>,
    learning_rate: f64,
}

pub enum Activation {
    ReLU,
    Softmax,
}

// enum wrapper for the activation functions so they can share common traits and methods
enum ActivationType {
    ReLU(ActivationReLU),
    Softmax(ActivationSoftmax),
}  


pub trait ActivationTrait {
    fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64>;
} 

impl NeuralNetwork {
    pub fn new(layers: Vec<(LayerDense, Activation)>, learning_rate: f64) -> Self {
        Self {
            layers,
            learning_rate,
        }
    }

    pub fn train(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        let mut output = inputs;
        for (layer, activiation) in &mut self.layers {
            layer.forward(output);
            let mut activation: ActivationType = match activiation {
                Activation::ReLU => ActivationType::ReLU(ActivationReLU::new()),
                Activation::Softmax => ActivationType::Softmax(ActivationSoftmax::new()),
                _ => panic!("Not implemented"),
            };

            match activation {
                ActivationType::ReLU(ref mut a) => output = a.forward(layer.output.clone()),
                ActivationType::Softmax(ref mut a) => output = a.forward(layer.output.clone()),
                _ => panic!("Not implemented"),
            }
        }

        let mut loss = Loss::new();
        loss.calculate(arr1(&[1., 0., 0.]), output.clone());
        output
    }
}