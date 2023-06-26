use rust_tensorflow_mini::NeuralNetworkTrait;
use ndarray::prelude::*;
use rust_tensorflow_mini::{LayerDense, ActivationReLU, ActivationSoftmax, ActivationTrait, spiral_data, OptimizerSGD, CategoricalCrossEntropy, LossFunction, Dataloader};
use serde::{Serialize, Deserialize};

fn main() {

    // STEP 1: define a model

    // start by defining a model with layers and activation functions
    #[derive(Serialize, Deserialize)] // <-- this is needed for saving and loading the model otherwise it will throw an error
    struct MyModel {
        l1: LayerDense,
        a1: ActivationReLU,
        l2: LayerDense,
        a2: ActivationSoftmax,
    }

    // using the neural network trait, we can define the forward pass
    impl NeuralNetworkTrait<Array2<f64>> for MyModel {
        type Model = MyModel;
        fn new (input_size: i32, output_size: i32) -> Self {

            let l1 = LayerDense::new(input_size, 64);
            let a1 = ActivationReLU::new();
            let l2 = LayerDense::new(64, output_size);
            let a2 = ActivationSoftmax::new();
            
            Self {
                l1,
                a1,
                l2,
                a2,
            }
        }
        fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
            
            self.l1.forward(inputs);
            self.a1.forward(self.l1.output.clone());
            self.l2.forward(self.a1.output.clone());
            self.a2.forward(self.l2.output.clone());
            
            self.a2.output.clone()

        }

    }

    // STEP 2: define the dataset

    let data = spiral_data(100, 3);
    let x = data.0.clone();
    let y = data.1.into_shape([1, 300]).unwrap();

    let dataloader = Dataloader::new(x.clone(), y.clone(), 300);

    println!("{:?} {:?} {:?}", dataloader.batches.len(), dataloader.batches[0].0.shape(), dataloader.batches[0].1.shape());

    println!("{:?}", x.shape());
    println!("{:?}", y.shape());


    println!("{:?}", dataloader.batches.len());

    // STEP 3: Training the model
    
    // #3.1 Hyperparameters

    const EPOCHS: usize = 10000;
    const LEARNING_RATE: f64 = 0.001;

    // #3.2 Initialize the model

    let mut model = MyModel::new(2, 4);

    //model.save("./data/model.txt".to_string()).expect("Could not save model");

    //model.load("./data/model.txt".to_string()).expect("Could not load model");

    let params = model.params();
    println!("{:?}", params);

    panic!("Model saved");

    // #3.3 Train the model

    let optmizer = OptimizerSGD::new(LEARNING_RATE);
    
    for epoch in 0..EPOCHS {

        for (batch_ix, (x, target) ) in dataloader.clone().into_iter().enumerate(){

            // forward pass
            let output = model.forward(x.clone());
            // calculate stats //TODO make a seperate function for this and rename calculate to forward
            let stats = CategoricalCrossEntropy::calculate(output.clone(), target.clone());
            
            let loss = stats.0;
            let acc = stats.1;
            
            // backward pass
            let dinputs = CategoricalCrossEntropy::backward(output.clone(), y.clone());
            model.l2.backward(dinputs.clone());
            model.a1.backward(model.l2.dinputs.clone());
            model.l1.backward(model.a1.dinputs.clone());

            // update weights and biases
            optmizer.update_params(&mut model.l1);
            optmizer.update_params(&mut model.l2);
            // print loss and accuracy
            if batch_ix % 5 == 0 {
                println!("Batch idx: {}", batch_ix);
                println!("LOSS: {:?}", loss);
                println!("ACCURACY: {}", acc);
                println!("\n");
            }
            
        }

        if epoch % 100 == 0 {
            println!("#############");
            println!("# EPOCH: {} #", epoch);
            println!("#############");
        }

    }

}