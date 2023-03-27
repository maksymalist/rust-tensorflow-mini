#![allow(unused)] // For beginning only.

use crate::prelude::*;
use std::fs::read_dir;
use ndarray::prelude::*;
use utils::{LayerDense, spiral_data, scatter_plot, NeuralNetwork, Activation, Loss, ActivationSoftmax, ActivationReLU};

mod error;
mod prelude;
mod utils;

fn main() -> Result<()> {

	let x = array![
		[1., 2., 3., 2.5],
		[2., 5., -1., 2.],
		[-1.5, 2.7, 3.3, -0.8]
	];

	let data = spiral_data(100, 3);
	let a = data.0;
	let b = data.1;

	println!("a: {} \n\n\n", a);

	let mut nn = NeuralNetwork::new(
		vec![
			(LayerDense::new(2, 3), Activation::ReLU),
			(LayerDense::new(3, 3), Activation::Softmax),
		],
		1.2,
	);

	let out = nn.train(a.clone());
	println!("output: {}", out);


	// plot the data

    let mut v: Vec<Vec<f64>> = a
        .axis_iter(Axis(0))
        .map(|row| row.to_vec())
        .collect();

	for (i, class) in b.into_iter().enumerate() {
		v[i].push(class);
	}

	scatter_plot(v);

	Ok(())
}