//! Optimization algorithms for training neural networks
//!
//! This module provides various optimizers that update network parameters
//! during training to minimize the loss function.

mod sgd;

pub use sgd::OptimizerSGD;

use crate::layers::DenseLayer;

/// Trait defining the interface for all optimizers
///
/// Optimizers are responsible for updating layer parameters (weights and biases)
/// based on computed gradients.
pub trait Optimizer {
    /// Called before updating parameters (e.g., to update learning rate)
    fn pre_update_params(&mut self);

    /// Updates the parameters of a dense layer
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer whose parameters will be updated
    fn update_params(&self, layer: &mut DenseLayer);

    /// Called after updating parameters (e.g., to increment iteration counter)
    fn post_update_params(&mut self);

    /// Returns the current learning rate
    fn current_learning_rate(&self) -> f64;
}
