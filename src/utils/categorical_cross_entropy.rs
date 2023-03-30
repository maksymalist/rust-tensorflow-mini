use ndarray::prelude::*;
use ndarray::Zip;
use std::any::Any;
use std::f64::consts::E;
use crate::utils::LossFunction;

pub struct CategoricalCrossEntropy;

impl LossFunction for CategoricalCrossEntropy {

    fn calculate(y_pred: Array2<f64>, y_true: Array2<f64>) -> (f64, f64) {

        let samples = y_true.shape()[0];
        let y_pred_clipped = y_pred.mapv(|x| {
            if x > 1.0 - 1e-7 {
                1.0 - 1e-7
            } else if x < 1e-7 {
                1e-7
            } else {
                x
            }
        });

        let mut average_loss = 0.0;
        if y_true.shape()[0] == 1 {
           let mut vals: Vec<f64> = Vec::new();
           for (targ_idx, distribution) in y_true.iter().zip(y_pred_clipped.outer_iter()) {
                let i = *targ_idx as usize;
                let l: f64 = distribution[i];
                vals.push(-l.ln());
           };

           average_loss = Array1::from(vals).mean().unwrap_or(3.0);
        }
        // this is for one-hot encoded labels
        else {
            let mut vals: Vec<f64> = Vec::new();
            for (targ_idx, distribution) in y_true.outer_iter().zip(y_pred_clipped.outer_iter()) {
                let i = targ_idx.iter().position(|&x| x == 1.0).unwrap();
                let l: f64 = distribution[i];
                vals.push(-l.ln());
            }

            average_loss = Array1::from(vals).mean().unwrap_or(3.0);
        }

        // TODO - implement accuracy calculation
        let predictions = y_pred_clipped.map_axis(Axis(1), |row| {
            let mut max = 0.0;
            let mut max_idx = 0;
            for (i, &val) in row.iter().enumerate() {
                if val > max {
                    max = val;
                    max_idx = i;
                }
            }
            max_idx as f64
        });

        let mut accuracy: f64 = 0.0;

        if y_true.shape()[0] == 1 {

            let mut correct = Vec::new();

            for (pred, targ) in predictions.iter().zip(y_true.iter()) {
                if pred == targ {
                    correct.push(1);
                } else {
                    correct.push(0);
                }
            }
            accuracy = Array1::from(correct.clone()).sum() as f64 / correct.len() as f64;
        } else {
            let targets = y_true.map_axis(Axis(1), |row| {
                let mut max = 0.0;
                let mut max_idx = 0;
                for (i, &val) in row.iter().enumerate() {
                    if val > max {
                        max = val;
                        max_idx = i;
                    }
                }
                max_idx as f64
            });
            
            let mut correct = Vec::new();

            for (pred, targ) in predictions.iter().zip(targets.iter()) {
                if pred == targ {
                    correct.push(1);
                } else {
                    correct.push(0);
                }
            }

            accuracy = Array1::from(correct.clone()).sum() as f64 / correct.len() as f64;
        }

        (average_loss, accuracy)

    }
}