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

fn main() {
    // Notice that the shape for the inputs and Neuron's weights match
    let inputs = Array2::random((6, 4), Uniform::new(-4.5, 1.53));
    let mut dense_layer = DenseLayer::new(4, 3);

    println!("Layer: {:?}", dense_layer);
    dense_layer.forward(inputs);

    println!("Output: {:?}", dense_layer.outputs);
}
