//! Activation functions for neural networks
//!
//! This module provides various activation functions used to introduce
//! non-linearity into neural networks.

mod relu;
mod softmax;

pub use relu::ActivationReLU;
pub use softmax::ActivationSoftMax;

use ndarray::Array2;

/// Trait defining the interface for activation functions
///
/// All activation functions must implement forward and backward propagation
/// to be compatible with the training process.
pub trait Activation {
    /// Applies the activation function to the inputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input data as a 2D array
    fn forward(&mut self, inputs: &Array2<f64>);

    /// Computes the gradient of the activation function
    ///
    /// # Arguments
    ///
    /// * `dvalues` - Gradient from the next layer
    ///
    /// # Returns
    ///
    /// Gradient with respect to the inputs
    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64>;

    /// Returns the output from the last forward pass
    fn outputs(&self) -> &Array2<f64>;
}
