//! Layer implementations for neural networks
//!
//! This module provides various layer types that can be used to build neural networks.
//! Currently implements dense (fully connected) layers with forward and backward propagation.

mod dense;

pub use dense::DenseLayer;

use ndarray::Array2;

/// Trait defining the interface for all layer types
///
/// This trait ensures consistency across different layer implementations and
/// enables polymorphic behavior when building complex networks.
pub trait Layer {
    /// Performs forward propagation through the layer
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input data as a 2D array (batch_size × input_features)
    fn forward(&mut self, inputs: &Array2<f64>);

    /// Performs backward propagation through the layer
    ///
    /// # Arguments
    ///
    /// * `inputs` - Original input data from forward pass
    /// * `dvalues` - Gradient from the next layer
    fn backward(&mut self, inputs: &Array2<f64>, dvalues: &Array2<f64>);

    /// Returns the output of the layer from the last forward pass
    fn outputs(&self) -> &Array2<f64>;

    /// Returns the gradient with respect to inputs from the last backward pass
    fn dinputs(&self) -> &Array2<f64>;
}
