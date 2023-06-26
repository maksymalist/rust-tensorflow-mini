use rust_tensorflow_mini::{spiral_data, Dataloader};

#[test]
fn test_layer_dense() {
    // Create data
    let data = spiral_data(100, 3);
    let x = data.0.clone();
    let y = data.1.into_shape([1, 300]).unwrap();

    let dataloader = Dataloader::new(x.clone(), y.clone(), 30);

    assert_eq!(dataloader.batches.len(), 10);
}