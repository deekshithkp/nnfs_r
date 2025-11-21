//! `ReLU` (Rectified Linear Unit) activation function

use ndarray::{Array2, Zip};

use super::Activation;

/// Rectified Linear Unit (`ReLU`) activation function
///
/// `ReLU` is defined as f(x) = max(0, x). It's one of the most commonly used
/// activation functions in deep learning due to its simplicity and effectiveness.
///
/// # Properties
///
/// - Non-linear activation
/// - Computationally efficient
/// - Helps mitigate vanishing gradient problem
/// - Can suffer from "dying `ReLU`" problem
///
/// # Examples
///
/// ```
/// use nnfs_r::activations::{ActivationReLU, Activation};
/// use ndarray::arr2;
///
/// let mut relu = ActivationReLU::new();
/// let inputs = arr2(&[[1.0, -2.0, 3.0]]);
/// relu.forward(&inputs);
/// // Output will be [[1.0, 0.0, 3.0]]
/// ```
pub struct ActivationReLU {
    /// Stored inputs from forward pass (needed for backward pass)
    inputs: Array2<f64>,
    /// Output after applying `ReLU`
    outputs: Array2<f64>,
}

impl ActivationReLU {
    /// Creates a new `ReLU` activation function
    pub fn new() -> Self {
        ActivationReLU {
            inputs: Array2::zeros((0, 0)),
            outputs: Array2::zeros((0, 0)),
        }
    }
}

impl Default for ActivationReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for ActivationReLU {
    /// Applies `ReLU`: output = max(0, input)
    fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs.clone_from(inputs);
        self.outputs = inputs.mapv(|x| x.max(0.0));
    }

    /// Computes gradient: derivative is 1 if input > 0, else 0
    fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
        let mut dinputs = dvalues.clone();
        Zip::from(&self.inputs)
            .and(&mut dinputs)
            .for_each(|&input, dinput| {
                if input <= 0.0 {
                    *dinput = 0.0;
                }
            });
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
    fn test_relu_forward() {
        let mut relu = ActivationReLU::new();
        let inputs = arr2(&[[1.0, -2.0, 3.0, -4.0], [0.5, -0.5, 0.0, 2.0]]);
        relu.forward(&inputs);

        assert!((relu.outputs[[0, 0]] - 1.0).abs() < f64::EPSILON);
        assert!((relu.outputs[[0, 1]] - 0.0).abs() < f64::EPSILON);
        assert!((relu.outputs[[0, 2]] - 3.0).abs() < f64::EPSILON);
        assert!((relu.outputs[[0, 3]] - 0.0).abs() < f64::EPSILON);
        assert!((relu.outputs[[1, 0]] - 0.5).abs() < f64::EPSILON);
        assert!((relu.outputs[[1, 1]] - 0.0).abs() < f64::EPSILON);
        assert!((relu.outputs[[1, 2]] - 0.0).abs() < f64::EPSILON);
        assert!((relu.outputs[[1, 3]] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = ActivationReLU::new();
        let inputs = arr2(&[[1.0, -2.0], [0.0, 3.0]]);
        relu.forward(&inputs);

        let dvalues = arr2(&[[1.0, 1.0], [1.0, 1.0]]);
        let dinputs = relu.backward(&dvalues);

        assert!((dinputs[[0, 0]] - 1.0).abs() < f64::EPSILON); // input > 0, gradient passes through
        assert!((dinputs[[0, 1]] - 0.0).abs() < f64::EPSILON); // input < 0, gradient is 0
        assert!((dinputs[[1, 0]] - 0.0).abs() < f64::EPSILON); // input = 0, gradient is 0
        assert!((dinputs[[1, 1]] - 1.0).abs() < f64::EPSILON); // input > 0, gradient passes through
    }
}
