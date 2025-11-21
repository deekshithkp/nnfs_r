//! Categorical cross-entropy loss function

use ndarray::{Array1, Array2, Axis};

use super::Loss;

/// Categorical Cross-Entropy Loss
///
/// This loss function is used for multi-class classification problems
/// where each sample belongs to exactly one class.
///
/// # Formula
///
/// L = -1/N * Σ log(p_c) where p_c is the predicted probability for the correct class
///
/// # Properties
///
/// - Works well with softmax activation in the output layer
/// - Outputs are always positive
/// - Lower values indicate better predictions
/// - Penalizes confident wrong predictions heavily
///
/// # Examples
///
/// ```
/// use nnfs_r::losses::{CategoricalCrossEntropyLoss, Loss};
/// use ndarray::{arr2, arr1};
///
/// let predictions = arr2(&[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]);
/// let labels = arr1(&[0, 1]);
/// let loss = CategoricalCrossEntropyLoss::forward(&predictions, &labels);
/// assert!(loss > 0.0);
/// ```
pub struct CategoricalCrossEntropyLoss;

impl CategoricalCrossEntropyLoss {
    /// Creates a new categorical cross-entropy loss function
    pub fn new() -> Self {
        CategoricalCrossEntropyLoss {}
    }
}

impl Default for CategoricalCrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CategoricalCrossEntropyLoss {
    /// Computes categorical cross-entropy loss
    ///
    /// Clips predictions to prevent log(0) and ensures numerical stability
    fn forward(y_pred: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
        let samples = y_pred.len_of(Axis(0));

        // Clip values to prevent log(0) and log(1)
        let clipped_values = y_pred.mapv(|x| x.clamp(f64::EPSILON, 1.0 - f64::EPSILON));

        // Extract confidence values for correct classes
        let correct_confidences = y_true
            .iter()
            .enumerate()
            .map(|(i, &label)| clipped_values[[i, label]])
            .collect::<Array1<f64>>();

        // Compute negative log likelihoods
        let negative_log_likelihoods = correct_confidences.mapv(|v| -v.ln());

        // Return mean loss
        negative_log_likelihoods.sum() / samples as f64
    }

    /// Computes gradient of categorical cross-entropy loss
    ///
    /// Converts labels to one-hot encoding and applies the gradient formula
    fn backward(dvalues: &Array2<f64>, y_true: &Array1<usize>) -> Array2<f64> {
        let samples = dvalues.shape()[0];
        let labels = dvalues.shape()[1];

        // Convert labels to one-hot encoding
        let mut y_true_one_hot = Array2::<f64>::zeros((samples, labels));
        for (i, &label) in y_true.iter().enumerate() {
            y_true_one_hot[[i, label]] = 1.0;
        }

        // Compute gradient: -y_true / y_pred, normalized by batch size
        let dinputs = -(&y_true_one_hot / dvalues) / samples as f64;
        dinputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_categorical_crossentropy_forward() {
        // Perfect predictions should have very low loss
        let perfect_pred = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let labels = arr1(&[0, 1]);
        let loss = CategoricalCrossEntropyLoss::forward(&perfect_pred, &labels);
        assert!(loss < 0.01); // Should be very close to 0

        // Poor predictions should have higher loss
        let poor_pred = arr2(&[[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]);
        let poor_loss = CategoricalCrossEntropyLoss::forward(&poor_pred, &labels);
        assert!(poor_loss > loss);
    }

    #[test]
    fn test_categorical_crossentropy_backward() {
        let predictions = arr2(&[[0.7, 0.2, 0.1], [0.1, 0.5, 0.4]]);
        let labels = arr1(&[0, 1]);
        let gradients = CategoricalCrossEntropyLoss::backward(&predictions, &labels);

        assert_eq!(gradients.shape(), predictions.shape());
        // Gradients should be finite
        assert!(gradients.iter().all(|&x| x.is_finite()));
    }
}
