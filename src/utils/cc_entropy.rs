use ndarray::prelude::*;

pub struct Loss {
    pub loss: f32,
    pub entropy: f32,
}

impl Loss {
    pub fn new() -> Self {
        Self {
            loss: 0.0,
            entropy: 0.0,
        }
    }

    pub fn calculate(&mut self) {
        let softmax_output = array![0.7, 0.1, 0.2];
        let target_output = array![1.0, 0.0, 0.0];

        let loss = -((target_output * softmax_output.mapv(f32::ln)).sum());
        println!("Loss: {}", loss);
    }

    fn forward(&mut self, inputs: Array2<f64>) {
        // black box
    }
}