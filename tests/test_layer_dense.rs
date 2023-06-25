use ndarray::array;
use rust_tensorflow_mini::LayerDense;

#[test]
fn test_layer_dense() {
    // Create a new layer with 3 input features and 5 output features
    let mut layer1 = LayerDense::new(3, 5);
    let mut layer2 = LayerDense::new(5, 2);

    let inputs = array![[1., 2., 3.], [4., 5., 6.]];

    layer1.forward(inputs);
    layer2.forward(layer1.output.clone());

    assert_eq!(layer1.output.shape(), [2, 5]);
    assert_eq!(layer2.output.shape(), [2, 2]);
}