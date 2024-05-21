use ndarray::Array2;
use ndarray_rand::{self, rand_distr::Uniform, RandomExt};


#[derive(Debug)]
struct DenseLayer {
    weights: Array2<f64>,
    biases: Array2<f64>,
    outputs: Array2<f64>,
}

impl DenseLayer {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        DenseLayer {
            weights: Array2::random((n_inputs, n_neurons), Uniform::new(0.0, 0.01)),
            biases: Array2::zeros((1, n_neurons)),
            outputs: Array2::zeros((n_inputs, 1))
        }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) {
        self.outputs = inputs.dot(&self.weights) + &self.biases;
    }
}

struct ActivationReLU {
    outputs: Array2<f64>,
}

impl ActivationReLU {
    pub fn new() -> Self {
        ActivationReLU {
            outputs: Array2::zeros((0, 0)),
        }
    }

    // ReLU is a linear function that outputs the input for all + inputs and 0 for -ve inputs
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = inputs.mapv(|i| i.max(0.0));
    }
}

fn main() {
    // Notice that the shape for the inputs and Neuron's weights match
    // Random data is fine for now but we would need to have the sample data in a persistent storage (likely a file) when we get to training the network
    let inputs = Array2::random((6, 4), Uniform::new(-2.5, 2.53));
    let mut dense_layer = DenseLayer::new(4, 3);

    println!("Layer: {:?}", dense_layer);
    dense_layer.forward(inputs);

    println!("Output: {:?}", dense_layer.outputs);

    let mut activation_relu = ActivationReLU::new();
    activation_relu.forward(&dense_layer.outputs);
    println!("ReLU Output: {:?}", activation_relu.outputs);
}
