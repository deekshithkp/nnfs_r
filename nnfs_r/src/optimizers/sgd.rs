//! Stochastic Gradient Descent (SGD) optimizer

use crate::layers::DenseLayer;
use super::Optimizer;

/// Stochastic Gradient Descent optimizer
///
/// SGD is a fundamental optimization algorithm that updates parameters
/// in the opposite direction of the gradient to minimize loss.
///
/// # Features
///
/// - Learning rate decay: reduces learning rate over time
/// - Momentum: accumulates velocity to accelerate learning and reduce oscillations
///
/// # Formula
///
/// Without momentum:
/// - weight = weight - learning_rate * gradient
///
/// With momentum:
/// - velocity = momentum * velocity - learning_rate * gradient
/// - weight = weight + velocity
///
/// # Examples
///
/// ```
/// use nnfs_r::optimizers::OptimizerSGD;
/// use nnfs_r::layers::DenseLayer;
///
/// // Create optimizer with learning rate 0.1, no decay, no momentum
/// let mut optimizer = OptimizerSGD::new(0.1, 0.0, 0.0);
///
/// // With momentum and decay
/// let mut optimizer_advanced = OptimizerSGD::new(0.1, 1e-3, 0.9);
/// ```
pub struct OptimizerSGD {
    /// Initial learning rate
    learning_rate: f64,
    /// Current learning rate (after decay)
    current_learning_rate: f64,
    /// Learning rate decay factor
    decay: f64,
    /// Current training iteration
    iteration: usize,
    /// Momentum coefficient (0.0 to 1.0)
    momentum: f64,
}

impl OptimizerSGD {
    /// Creates a new SGD optimizer
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Initial learning rate (typically 0.001 to 0.1)
    /// * `decay` - Learning rate decay (0.0 for no decay, typical values: 1e-3 to 1e-5)
    /// * `momentum` - Momentum coefficient (0.0 for no momentum, typical values: 0.9 to 0.99)
    ///
    /// # Returns
    ///
    /// A new `OptimizerSGD` instance
    pub fn new(learning_rate: f64, decay: f64, momentum: f64) -> Self {
        OptimizerSGD {
            learning_rate,
            current_learning_rate: learning_rate,
            decay,
            iteration: 0,
            momentum,
        }
    }

    /// Creates an SGD optimizer with default parameters
    ///
    /// Default: learning_rate=0.01, decay=0.0, momentum=0.0
    pub fn default_params() -> Self {
        Self::new(0.01, 0.0, 0.0)
    }

    /// Creates an SGD optimizer with momentum
    ///
    /// Recommended parameters for general use
    pub fn with_momentum() -> Self {
        Self::new(0.01, 1e-3, 0.9)
    }
}

impl Optimizer for OptimizerSGD {
    /// Updates learning rate based on decay
    ///
    /// Formula: lr = initial_lr * (1 / (1 + decay * iteration))
    fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate =
                self.learning_rate * (1.0 / (1.0 + self.decay * self.iteration as f64));
        }
    }

    /// Updates layer parameters using gradients
    fn update_params(&self, layer: &mut DenseLayer) {
        if self.momentum != 0.0 {
            // Update with momentum
            let weight_updates =
                self.momentum * layer.weight_momentums.clone() - self.current_learning_rate * &layer.dweights;
            layer.weight_momentums = weight_updates.clone();

            let bias_updates =
                self.momentum * layer.bias_momentums.clone() - self.current_learning_rate * &layer.dbiases;
            layer.bias_momentums = bias_updates.clone();

            layer.weights += &weight_updates;
            layer.biases += &bias_updates;
        } else {
            // Standard SGD update
            layer.weights = layer.weights.clone() - self.current_learning_rate * &layer.dweights;
            layer.biases = layer.biases.clone() - self.current_learning_rate * &layer.dbiases;
        }
    }

    /// Increments iteration counter
    fn post_update_params(&mut self) {
        self.iteration += 1;
    }

    fn current_learning_rate(&self) -> f64 {
        self.current_learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_sgd_creation() {
        let optimizer = OptimizerSGD::new(0.1, 0.0, 0.0);
        assert_eq!(optimizer.current_learning_rate, 0.1);
        assert_eq!(optimizer.iteration, 0);
    }

    #[test]
    fn test_learning_rate_decay() {
        let mut optimizer = OptimizerSGD::new(1.0, 0.1, 0.0);
        
        optimizer.pre_update_params();
        assert_eq!(optimizer.current_learning_rate, 1.0); // First iteration: 1.0 / (1 + 0.1*0)
        
        optimizer.post_update_params();
        optimizer.pre_update_params();
        assert!((optimizer.current_learning_rate - 0.909).abs() < 0.01); // Second: 1.0 / (1 + 0.1*1)
    }

    #[test]
    fn test_parameter_update_without_momentum() {
        let mut layer = DenseLayer::new(2, 3);
        layer.weights = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        layer.biases = arr1(&[0.1, 0.2, 0.3]);
        layer.dweights = arr2(&[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]);
        layer.dbiases = arr1(&[0.1, 0.1, 0.1]);

        let optimizer = OptimizerSGD::new(0.1, 0.0, 0.0);
        optimizer.update_params(&mut layer);

        // weights should decrease by lr * gradient
        assert!((layer.weights[[0, 0]] - 0.99).abs() < 1e-10);
        assert!((layer.biases[0] - 0.09).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_update_with_momentum() {
        let mut layer = DenseLayer::new(2, 2);
        layer.weights = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        layer.biases = arr1(&[0.1, 0.2]);
        layer.dweights = arr2(&[[0.1, 0.1], [0.1, 0.1]]);
        layer.dbiases = arr1(&[0.1, 0.1]);

        let optimizer = OptimizerSGD::new(0.1, 0.0, 0.9);
        optimizer.update_params(&mut layer);

        // First update: momentum is 0, so should be same as without momentum
        assert!((layer.weights[[0, 0]] - 0.99).abs() < 1e-10);
        
        // Momentum should be updated
        assert!((layer.weight_momentums[[0, 0]] - (-0.01)).abs() < 1e-10);
    }
}
