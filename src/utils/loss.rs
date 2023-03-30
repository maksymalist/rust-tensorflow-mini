use ndarray::prelude::*;

pub trait LossFunction {
    fn calculate(y_pred: Array2<f64>, y_true: Array2<f64>) -> (f64, f64);
}
