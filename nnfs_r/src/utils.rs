//! Utility functions for neural network training and evaluation

use ndarray::{Array1, Array2};
use rand::Rng;

/// Generates spiral dataset for classification
///
/// Creates a synthetic dataset with multiple classes arranged in spiral patterns.
/// This is a common dataset for demonstrating neural network capabilities.
///
/// # Arguments
///
/// * `samples` - Number of samples per class
/// * `classes` - Number of classes to generate
///
/// # Returns
///
/// A tuple containing:
/// - `X`: Feature matrix of shape (samples * classes, 2)
/// - `y`: Label vector of shape (samples * classes,)
///
/// # Examples
///
/// ```
/// use nnfs_r::utils::create_data;
///
/// let (X, y) = create_data(100, 3);
/// assert_eq!(X.shape(), &[300, 2]);
/// assert_eq!(y.len(), 300);
/// ```
pub fn create_data(samples: usize, classes: usize) -> (Array2<f64>, Array1<usize>) {
    let mut rng = rand::thread_rng();
    let mut X = Array2::<f64>::zeros((samples * classes, 2));
    let mut y = Array1::<usize>::zeros(samples * classes);

    for class_number in 0..classes {
        for sample in 0..samples {
            let ix = sample + class_number * samples;
            let r = rng.gen_range(0.0..1.0);
            let t = rng.gen_range(class_number as f64 * 4.0..(class_number + 1) as f64 * 4.0)
                + r * 0.2;
            X[[ix, 0]] = r * t.sin();
            X[[ix, 1]] = r * t.cos();
            y[ix] = class_number;
        }
    }

    (X, y)
}

/// Computes classification accuracy
///
/// Calculates the percentage of correct predictions.
///
/// # Arguments
///
/// * `y_pred` - Predicted class labels
/// * `y_true` - Ground truth class labels
///
/// # Returns
///
/// Accuracy as a value between 0.0 and 1.0
///
/// # Examples
///
/// ```
/// use nnfs_r::utils::accuracy;
/// use ndarray::arr1;
///
/// let predictions = arr1(&[0, 1, 2, 1]);
/// let labels = arr1(&[0, 1, 1, 1]);
/// let acc = accuracy(&predictions, &labels);
/// assert_eq!(acc, 0.75); // 3 out of 4 correct
/// ```
pub fn accuracy(y_pred: &Array1<usize>, y_true: &Array1<usize>) -> f64 {
    y_pred
        .iter()
        .zip(y_true.iter())
        .filter(|&(a, b)| a == b)
        .count() as f64
        / y_true.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_create_data() {
        let (X, y) = create_data(50, 3);
        
        assert_eq!(X.shape(), &[150, 2]); // 50 samples * 3 classes = 150 total
        assert_eq!(y.len(), 150);
        
        // Check that labels are in valid range
        assert!(y.iter().all(|&label| label < 3));
        
        // Check that each class has 50 samples
        for class in 0..3 {
            let count = y.iter().filter(|&&label| label == class).count();
            assert_eq!(count, 50);
        }
    }

    #[test]
    fn test_accuracy_perfect() {
        let predictions = arr1(&[0, 1, 2, 0, 1, 2]);
        let labels = arr1(&[0, 1, 2, 0, 1, 2]);
        assert_eq!(accuracy(&predictions, &labels), 1.0);
    }

    #[test]
    fn test_accuracy_partial() {
        let predictions = arr1(&[0, 1, 2, 1]);
        let labels = arr1(&[0, 1, 1, 1]);
        assert_eq!(accuracy(&predictions, &labels), 0.75);
    }

    #[test]
    fn test_accuracy_zero() {
        let predictions = arr1(&[0, 0, 0, 0]);
        let labels = arr1(&[1, 1, 1, 1]);
        assert_eq!(accuracy(&predictions, &labels), 0.0);
    }
}
