use crate::prelude::*;

//neural network
mod neural_network;
pub use neural_network::*;

// layer dense
mod layer_dense;
pub use layer_dense::*;

// activation relu
mod relu;
pub use relu::*;

// activation softmax
mod softmax;
pub use softmax::*;

// loss function
mod cc_entropy;
pub use cc_entropy::*;

// data
mod data;
pub use data::*;

// visualize
mod visualize;
pub use visualize::*;