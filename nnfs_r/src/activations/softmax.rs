//! Softmax activation function

use ndarray::{s, Array, Array2, Axis};
use ndarray_stats::QuantileExt;

use super::Activation;

/// Softmax activation function
///
/// Softmax converts a vector of values into a probability distribution.
/// It's commonly used in the output layer for multi-class classification.
///
/// # Formula
///
/// For input vector x, softmax(x)_i = exp(x_i) / Σ(exp(x_j))
///
/// # Properties
///
/// - Output values are in range (0, 1)
/// - Output values sum to 1 (valid probability distribution)
/// - Monotonic function
/// - Differentiable everywhere
///
/// # Examples
///
/// ```
/// use nnfs_r::activations::{ActivationSoftMax, Activation};
/// use ndarray::arr2;
///
/// let mut softmax = ActivationSoftMax::new();
/// let inputs = arr2(&[[1.0, 2.0, 3.0]]);
/// softmax.forward(&inputs);
/// // Output will be a probability distribution that sums to 1.0
/// ```
pub struct ActivationSoftMax {
    /// Output probability distribution
    pub(crate) outputs: Array2<f64>,
}

impl ActivationSoftMax {
    /// Creates a new Softmax activation function
    pub fn new() -> Self {
        ActivationSoftMax {
            outputs: Array2::zeros((0, 0)),
        }
    }
}

impl Default for ActivationSoftMax {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for ActivationSoftMax {
    /// Applies Softmax activation with numerical stability
    ///
    /// Uses the trick of subtracting max value to prevent overflow
    fn forward(&mut self, inputs: &Array2<f64>) {
        // Subtract max for numerical stability
        let max_values = inputs.map_axis(Axis(1), |row| *row.max().unwrap());
        let exp_values = (inputs - &max_values.insert_axis(Axis(1))).mapv(f64::exp);
        let sum_exp_values = exp_values.sum_axis(Axis(1));
        self.outputs = exp_values / &sum_exp_values.insert_axis(Axis(1));
    }

    /// Computes gradient using Jacobian matrix
    ///
    /// For each sample, computes the Jacobian matrix of softmax and
    /// applies it to the gradient from the next layer.
    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
        let mut dinputs = Array2::<f64>::zeros(dvalues.raw_dim());

        for (index, (single_output, single_dvalues)) in self
            .outputs
            .axis_iter(Axis(0))
            .zip(dvalues.axis_iter(Axis(0)))
            .enumerate()
        {
            // Reshape output to column vector for matrix multiplication
            let single_output = single_output.into_shape((single_output.len(), 1)).unwrap();
            
            // Compute Jacobian matrix: diag(output) - output * output^T
            let jacobian_matrix =
                Array::eye(single_output.len()) - &single_output.dot(&single_output.t());
            
            // Apply chain rule
            let gradient = jacobian_matrix.dot(&single_dvalues);
            dinputs.slice_mut(s![index, ..]).assign(&gradient);
        }

        dinputs
    }

    fn outputs(&self) -> &Array2<f64> {
        &self.outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_softmax_forward() {
        let mut softmax = ActivationSoftMax::new();
        let inputs = arr2(&[[1.0, 2.0, 3.0]]);
        softmax.forward(&inputs);

        // Check that outputs sum to 1.0
        let sum: f64 = softmax.outputs.sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        for &val in softmax.outputs.iter() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut softmax = ActivationSoftMax::new();
        // Large values that could cause overflow without max subtraction
        let inputs = arr2(&[[1000.0, 1001.0, 1002.0]]);
        softmax.forward(&inputs);

        // Should still produce valid probabilities
        let sum: f64 = softmax.outputs.sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(softmax.outputs.iter().all(|&x| x.is_finite()));
    }
}
