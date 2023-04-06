#![allow(unused)] // For beginning only.

use crate::prelude::*;
use std::fs::read_dir;
use ndarray::prelude::*;
use utils::{LayerDense, spiral_data, scatter_plot, NeuralNetwork, Activation, LossFunction, ActivationSoftmax, ActivationReLU, CategoricalCrossEntropy, ActivationTrait, OptimizerSDG};

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

	fn forward (x: Array2<f64>, y_true: Array2<f64>) {
		let mut dense1 = LayerDense::new(2, 64);
		let mut activation1 = ActivationReLU::new();
		let mut dense2 = LayerDense::new(64, 3);
		let mut softmax = ActivationSoftmax::new();

		let optimzer = OptimizerSDG::new(0.05);
		const EPOCHS: usize = 10001;

		for epoch in 0..EPOCHS {

			// forward pass
			dense1.forward(x.clone());
			activation1.forward(dense1.output.clone());
			dense2.forward(activation1.output.clone());
			softmax.forward(dense2.output.clone());

			let loss = CategoricalCrossEntropy::calculate(softmax.output.clone(), y_true.clone());

			// backward pass

			/*✅*/ let mut dinputs1 = CategoricalCrossEntropy::backward(softmax.output.clone(), y_true.clone());

			dense2.backward(dinputs1.clone());
			activation1.backward(dense2.dinputs.clone());
			dense1.backward(activation1.dinputs.clone());

			// println!("dense2: \n{:?}\n", dense2.dinputs.shape());
			// println!("activation1: \n{:?}\n", activation1.dinputs.shape());
			// println!("dense1: \n{:?}\n", dense1.dinputs.shape());

			// if epoch is a multiple of 100, print loss and accuracy

			if epoch % 1000 == 0 {

				println!("EPOCH: {}", epoch);
				println!("LOSS: {:?}", loss.0);
				println!("ACCURACY: {}", loss.1);

				println!("\n")
			}

			// println!("DWEIGHTS 1: {:?}", dense1.dweights);
			// println!("DBIASES 1: {:?}", dense1.dbiases);
			// println!("-------------------------------");
			// println!("DWEIGHTS 2: {:?}", dense2.dweights);
			// println!("DBIASES 2: {:?}", dense2.dbiases);

			// update weights
			optimzer.update_params(&mut dense1, epoch);
			optimzer.update_params(&mut dense2, epoch);
		}



	}

	forward(a.clone(), b.clone().into_shape([1, 300]).unwrap());

	// let mut nn = NeuralNetwork::new(
	// 	vec![
	// 		(LayerDense::new(2, 3), Activation::ReLU),
	// 		(LayerDense::new(3, 3), Activation::Softmax),
	// 	],
	// 	1.2,
	// );

	// let out = nn.train(a.clone(), b.clone().into_shape([1, 300]).unwrap());

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