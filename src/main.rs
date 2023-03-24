#![allow(unused)] // For beginning only.

use crate::{prelude::*, utils::ActivationReLU};
use std::fs::read_dir;
use ndarray::prelude::*;
use utils::{LayerDense, spiral_data, scatter_plot};

mod error;
mod prelude;
mod utils;

fn main() -> Result<()> {

	let x = array![
		[1., 2., 3., 2.5],
		[2., 5., -1., 2.],
		[-1.5, 2.7, 3.3, -0.8]
	];

	let mut layer1 = LayerDense::new(4, 5);
	println!("weights: {}", layer1.weights);
	let mut activation1 = ActivationReLU::new();
	layer1.forward(x);
	println!("output 1: {}", layer1.output);
	activation1.forward(layer1.output);
	// output of layer 1 has to be the same as the input of layer 2
	//let mut layer2 = LayerDense::new(5, 6);

	//layer2.forward(layer1.output);

	println!("output ac: {}", activation1.output); 

	let data = spiral_data(100, 3);
	let a = data.0;
	let b = data.1;

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