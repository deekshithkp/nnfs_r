//! Combined Softmax activation and Categorical Cross-Entropy loss
//!
//! This module provides an optimized implementation that combines
//! softmax activation with categorical cross-entropy loss.

use ndarray::{Array1, Array2};

use super::{CategoricalCrossEntropyLoss, Loss};
use crate::activations::{Activation, ActivationSoftMax};

/// Combined Softmax Activation and Categorical Cross-Entropy Loss
///
/// This implementation combines softmax activation and categorical cross-entropy
/// loss into a single component. This is more efficient than using them separately
/// because the backward pass can be simplified mathematically.
///
/// # Mathematical Optimization
///
/// When combining softmax and cross-entropy, the gradient simplifies to:
/// `d_loss/d_input` = (`softmax_output` - `one_hot_labels`) / `batch_size`
///
/// This is much simpler than computing gradients separately and is more
/// numerically stable.
///
/// # Examples
///
/// ```
/// use nnfs_r::losses::ActivationSoftMaxLossCategoricalCrossEntropy;
/// use ndarray::{arr2, arr1};
///
/// let mut loss_activation = ActivationSoftMaxLossCategoricalCrossEntropy::new();
/// let logits = arr2(&[[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]]);
/// let labels = arr1(&[0, 1]);
/// let loss = loss_activation.forward(&logits, &labels);
/// ```
pub struct ActivationSoftMaxLossCategoricalCrossEntropy {
    /// Underlying softmax activation
    activation: ActivationSoftMax,
}

impl ActivationSoftMaxLossCategoricalCrossEntropy {
    /// Creates a new combined activation and loss function
    pub fn new() -> Self {
        ActivationSoftMaxLossCategoricalCrossEntropy {
            activation: ActivationSoftMax::new(),
        }
    }

    /// Performs forward pass: applies softmax then computes loss
    ///
    /// # Arguments
    ///
    /// * `inputs` - Raw logits from the previous layer
    /// * `y_true` - Ground truth class labels
    ///
    /// # Returns
    ///
    /// The categorical cross-entropy loss value
    pub fn forward(&mut self, inputs: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
        self.activation.forward(inputs);
        CategoricalCrossEntropyLoss::forward(&self.activation.outputs, y_true)
    }

    /// Performs backward pass using the simplified gradient
    ///
    /// # Arguments
    ///
    /// * `dvalues` - Softmax outputs (typically same as from forward pass)
    /// * `y_true` - Ground truth class labels
    ///
    /// # Returns
    ///
    /// Gradient with respect to the input logits
    pub fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array1<usize>) -> Array2<f64> {
        let samples = dvalues.shape()[0];
        let classes = dvalues.shape()[1];

        // Convert labels to one-hot encoding
        let mut y_true_one_hot = Array2::<f64>::zeros((samples, classes));
        for (i, &label) in y_true.iter().enumerate() {
            y_true_one_hot[[i, label]] = 1.0;
        }

        // Simplified gradient for combined softmax + categorical cross-entropy
        
        (&self.activation.outputs - &y_true_one_hot) / samples as f64
    }

    /// Returns a reference to the softmax outputs from the last forward pass
    pub fn outputs(&self) -> &Array2<f64> {
        &self.activation.outputs
    }
}

impl Default for ActivationSoftMaxLossCategoricalCrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_combined_forward() {
        let mut combined = ActivationSoftMaxLossCategoricalCrossEntropy::new();
        let logits = arr2(&[[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]]);
        let labels = arr1(&[0, 1]);

        let loss = combined.forward(&logits, &labels);

        // Loss should be positive
        assert!(loss > 0.0);
        // Outputs should be valid probabilities
        let sum: f64 = combined.outputs().sum_axis(ndarray::Axis(1)).sum();
        assert!((sum - 2.0).abs() < 1e-6); // 2 samples, each row sums to 1
    }

    #[test]
    fn test_combined_backward() {
        let mut combined = ActivationSoftMaxLossCategoricalCrossEntropy::new();
        let logits = arr2(&[[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]]);
        let labels = arr1(&[0, 1]);

        combined.forward(&logits, &labels);
        let outputs = combined.outputs().clone();
        let gradients = combined.backward(&outputs, &labels);

        assert_eq!(gradients.shape(), logits.shape());
        // Gradients should sum to approximately 0 (due to normalization)
        let grad_sum: f64 = gradients.sum();
        assert!(grad_sum.abs() < 1e-10);
    }

    #[test]
    fn test_gradient_correctness() {
        // The gradient at the correct class should be (probability - 1) / batch_size
        let mut combined = ActivationSoftMaxLossCategoricalCrossEntropy::new();
        let logits = arr2(&[[10.0, 0.0, 0.0]]); // Very confident prediction for class 0
        let labels = arr1(&[0]);

        combined.forward(&logits, &labels);
        let outputs = combined.outputs().clone();
        let gradients = combined.backward(&outputs, &labels);

        // For very confident correct prediction, gradient should be close to (1.0 - 1.0) / 1 = 0
        assert!(gradients[[0, 0]].abs() < 0.01);
    }
}
