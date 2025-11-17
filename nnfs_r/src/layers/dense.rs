//! Dense (fully connected) layer implementation

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use super::Layer;

/// A dense (fully connected) layer in a neural network
///
/// This layer connects every input neuron to every output neuron through
/// learned weights and biases.
///
/// # Examples
///
/// ```
/// use nnfs_r::layers::{DenseLayer, Layer};
/// use ndarray::Array2;
///
/// let mut layer = DenseLayer::new(3, 5); // 3 inputs, 5 neurons
/// let inputs = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// layer.forward(&inputs);
/// ```
#[derive(Debug)]
pub struct DenseLayer {
    /// Weight matrix (n_inputs × n_neurons)
    pub weights: Array2<f64>,
    /// Bias vector (n_neurons)
    pub biases: Array1<f64>,
    /// Output from forward propagation
    pub outputs: Array2<f64>,
    /// Gradient of weights
    pub dweights: Array2<f64>,
    /// Gradient of biases
    pub dbiases: Array1<f64>,
    /// Gradient of inputs
    pub dinputs: Array2<f64>,
    /// Momentum for weights (used by optimizers)
    pub weight_momentums: Array2<f64>,
    /// Momentum for biases (used by optimizers)
    pub bias_momentums: Array1<f64>,
}

impl DenseLayer {
    /// Creates a new dense layer with random weight initialization
    ///
    /// Weights are initialized using a uniform distribution in the range [-0.01, 0.01]
    /// to break symmetry while keeping values small.
    ///
    /// # Arguments
    ///
    /// * `n_inputs` - Number of input features
    /// * `n_neurons` - Number of neurons in this layer
    ///
    /// # Returns
    ///
    /// A new `DenseLayer` instance
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        DenseLayer {
            weights: Array2::random((n_inputs, n_neurons), Uniform::new(-0.01, 0.01)),
            biases: Array1::zeros(n_neurons),
            outputs: Array2::zeros((0, 0)),
            dweights: Array2::zeros((n_inputs, n_neurons)),
            dbiases: Array1::zeros(n_neurons),
            dinputs: Array2::zeros((0, 0)),
            weight_momentums: Array2::zeros((n_inputs, n_neurons)),
            bias_momentums: Array1::zeros(n_neurons),
        }
    }
}

impl Layer for DenseLayer {
    /// Performs forward propagation: output = inputs · weights + biases
    fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = inputs.dot(&self.weights) + &self.biases;
    }

    /// Performs backward propagation to compute gradients
    ///
    /// Computes:
    /// - dweights = inputs^T · dvalues
    /// - dbiases = sum(dvalues, axis=0)
    /// - dinputs = dvalues · weights^T
    fn backward(&mut self, inputs: &Array2<f64>, dvalues: &Array2<f64>) {
        self.dweights = inputs.t().dot(dvalues);
        self.dbiases = dvalues.sum_axis(Axis(0));
        self.dinputs = dvalues.dot(&self.weights.t());
    }

    fn outputs(&self) -> &Array2<f64> {
        &self.outputs
    }

    fn dinputs(&self) -> &Array2<f64> {
        &self.dinputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_dense_layer_creation() {
        let layer = DenseLayer::new(3, 5);
        assert_eq!(layer.weights.shape(), &[3, 5]);
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn test_forward_propagation() {
        let mut layer = DenseLayer::new(2, 3);
        // Set known weights for testing
        layer.weights = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        layer.biases = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let inputs = arr2(&[[1.0, 2.0]]);
        layer.forward(&inputs);

        assert_eq!(layer.outputs.shape(), &[1, 3]);
        // Expected: [1.0*0.1 + 2.0*0.4 + 0.1, 1.0*0.2 + 2.0*0.5 + 0.2, 1.0*0.3 + 2.0*0.6 + 0.3]
        assert!((layer.outputs[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((layer.outputs[[0, 1]] - 1.4).abs() < 1e-10);
        assert!((layer.outputs[[0, 2]] - 1.8).abs() < 1e-10);
    }
}
