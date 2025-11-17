//! Loss functions for neural network training
//!
//! This module provides various loss (cost) functions used to measure
//! the difference between predicted and actual values during training.

mod categorical_crossentropy;
mod combined;

pub use categorical_crossentropy::CategoricalCrossEntropyLoss;
pub use combined::ActivationSoftMaxLossCategoricalCrossEntropy;

use ndarray::{Array1, Array2};

/// Trait defining the interface for loss functions
///
/// Loss functions measure how well the network's predictions match the
/// actual target values.
pub trait Loss {
    /// Computes the loss value
    ///
    /// # Arguments
    ///
    /// * `y_pred` - Predicted values from the network
    /// * `y_true` - Ground truth labels
    ///
    /// # Returns
    ///
    /// The average loss across all samples
    fn forward(y_pred: &Array2<f64>, y_true: &Array1<usize>) -> f64;

    /// Computes the gradient of the loss
    ///
    /// # Arguments
    ///
    /// * `dvalues` - Predicted values (typically same as y_pred from forward)
    /// * `y_true` - Ground truth labels
    ///
    /// # Returns
    ///
    /// Gradient with respect to predictions
    fn backward(dvalues: &Array2<f64>, y_true: &Array1<usize>) -> Array2<f64>;
}
