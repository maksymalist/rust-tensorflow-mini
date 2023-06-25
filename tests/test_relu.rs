use ndarray::prelude::*;

use rust_tensorflow_mini::LayerDense;
use rust_tensorflow_mini::ActivationReLU as Relu;
use rust_tensorflow_mini::ActivationTrait;

#[test]
fn test_relu() {
    let mut layer1 = LayerDense::new(3, 5);
    let mut layer2 = LayerDense::new(5, 2);
    let mut activation1 = Relu::new();
    let mut activation2 = Relu::new();

    let inputs = array![[1., 2., 3.], [4., 5., 6.]];

    layer1.forward(inputs);
    activation1.forward(layer1.output.clone());
    layer2.forward(activation1.output.clone());
    activation2.forward(layer2.output.clone());

    assert_eq!(layer1.output.shape(), [2, 5]);
    assert_eq!(activation1.output.shape(), [2, 5]);
    assert_eq!(layer2.output.shape(), [2, 2]);
    assert_eq!(activation2.output.shape(), [2, 2]);
}
