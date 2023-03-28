use ndarray::prelude::*;
use ndarray::Zip;
use std::f64::consts::E;

pub struct Loss {
    pub loss: f64,
}

impl Loss {
    pub fn new() -> Self {
        Self {
            loss: 0.0,
        }
    }

    pub fn calculate(&mut self, output: Array1<f64>, y: Array2<f64>) {
        let softmax_outputs = Array2::from_shape_vec((3, 3), vec![0.7, 0.1, 0.2, 0.1, 0.5, 0.4, 0.02, 0.9, 0.08]).unwrap();
        let class_targets = Array1::from(vec![0, 1, 1]);
        
        let mut vals: Vec<f64> = Vec::new();
        for (targ_idx, distribution) in class_targets.iter().zip(softmax_outputs.outer_iter()) {
            let i = *targ_idx as usize;
            let l: f64 = distribution[i];
            vals.push(-l.ln());
        }

        let loss = Array1::from(vals).mean().unwrap_or(3.0);
        self.loss = loss;
        
        
        self.forward(output.clone(), output.clone());
       println!("\n\n\n losst {:?} \n\n\n", loss);
    }

    fn forward(&mut self, y_pred: Array1<f64>, y_true: Array1<f64>) {
        let samples = y_true.shape()[0];
        let y_pred_clipped = y_pred.mapv(|x| x.max(1.0-1e-7).min(1e-7));  

        if y_true.shape()[0] == 1 {

        }
        //
    }
}