use ndarray::prelude::*;
use rand::Rng;

pub struct LayerDense {
    pub inputs: Array2<f64>,
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub output: Array2<f64>,
    pub dinputs: Array2<f64>,
    pub dweights: Array2<f64>,
    pub dbiases: Array1<f64>,
}


impl LayerDense {

    // this function initializes the weights (from -1.0 to 1) and biases (starting at 2.0) 
    pub fn new(n_inputs: i32, n_neurons: i32) -> Self {
        let mut rng = rand::thread_rng();

        Self { 
            // matrix of weights 
            /*
            weights: array![                        ]
                [0.2, 0.8, -0.5, 1.0], <-- 4 weights|
                [0.5, -0.91, 0.26, -0.5],           |
                [-0.26, -0.27, 0.17, 0.87]          |
                [0.5, -0.91, 0.26, -0.5],           |   5 neurons
                [-0.26, -0.27, 0.17, 0.87]          |
                [-0.26, -0.27, 0.17, 0.87]          |
                                                    ]
             */

            inputs: Array2::zeros((0, n_inputs as usize)),
            weights: Array2::from_shape_fn((n_neurons as usize, n_inputs as usize), |_| 0.10 * rng.gen_range(-1.0..1.0) as f64),

            // vector of biases for each neuron
            biases: Array1::from(vec![0.3; n_neurons as usize]),
            output: Array2::zeros((0, n_neurons as usize)),
            dbiases: Array1::zeros(n_neurons as usize),
            dweights: Array2::zeros((n_neurons as usize, n_inputs as usize)),
            dinputs: Array2::zeros((0, n_inputs as usize)),
        }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) {
        self.output = inputs.dot(&self.weights.t()) + &self.biases;
        self.inputs = inputs;
    }

    pub fn backward(&mut self, dvalues: Array2<f64>) {
        self.dweights = self.inputs.t().dot(&self.output);
        self.dbiases = dvalues.sum_axis(Axis(0));
        self.dinputs = dvalues.dot(&self.weights);
    }
}