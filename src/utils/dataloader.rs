use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub struct Dataloader {
    pub batches: Vec<(Array2<f64>, Array2<f64>)>,
    input: Array2<f64>,
    y_true: Array2<f64>,
    batch_size: i32,
}

impl Dataloader {
    pub fn new(input: Array2<f64>, y_true: Array2<f64>, batch_size: i32) -> Self {
        
        if input.shape()[0] as i32 % batch_size != 0 {
            panic!("Batch size must be a factor of the number of inputs");
        }

        let mut batches: Vec<(Array2<f64>, Array2<f64>)> = Vec::new();

        let mut batch_idx = 0;

        while batch_idx < input.shape()[0] / batch_size as usize {
            let x_batch = input.slice(s![batch_idx * batch_size as usize..(batch_idx + 1) * batch_size as usize, ..]);
            let y_true_batch = y_true.slice(s![batch_idx * batch_size as usize..(batch_idx + 1) * batch_size as usize, ..]);
            
            batches.push((x_batch.to_owned(), y_true_batch.to_owned()));
            batch_idx += 1;
        }

        Self {
            batches,
            input,
            batch_size,
            y_true,
        }
    }

}

impl IntoIterator for Dataloader {
    type Item = (Array2<f64>, Array2<f64>);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut batches: Vec<Self::Item> = Vec::new();
        for batch in self.batches.into_iter() {
            batches.push(batch.to_owned());
        }
        batches.into_iter()
    }
}