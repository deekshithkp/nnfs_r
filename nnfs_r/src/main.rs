use ndarray::{Axis,Array2};
use ndarray_rand::{self, rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;


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

struct ActivationSoftMax {
    outputs: Array2<f64>,
}

impl ActivationSoftMax {
    pub fn new() -> Self {
        ActivationSoftMax {
            outputs: Array2::zeros((0, 0)),
        }
    }

    // SoftMax is an exponential function that helps us classify the input into probabilities ideal for output layer
    // The input is exponentiated (only positive values), normalised (minimise variance) and grouped to classifications
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        // The exp value can skyrocket for large positive values, hence we calculate the max value for each row and subtract it from each value
        let max_values = inputs.map_axis(Axis(1), |row| *row.max().unwrap());
        
        // Subtract the max values from each row and compute the unnormalised exponentials
        // Notice that the max 1D array max_values is inserted with Axis 1 for compatibility
        let exp_values = inputs - &max_values.insert_axis(Axis(1));
        let exp_values = exp_values.mapv(f64::exp);

        // Compute the sum of exponentials along axis 1
        let sum_exp_values = exp_values.sum_axis(Axis(1));

        // Normalize probabilities for each sample
        let probabilities = exp_values / &sum_exp_values.insert_axis(Axis(1));

        // Store the result in the output
        self.outputs = probabilities;

    }
}

fn main() {
    // Notice that the shape for the inputs and Neuron's weights match
    // Random data is fine for now but we would need to have the sample data in a persistent storage (likely a file) when we get to training the network
    let inputs = Array2::random((6, 4), Uniform::new(-2.5, 2.53));
    let mut dense_layer1 = DenseLayer::new(4, 3);

    dense_layer1.forward(inputs);
    println!("Layer 1 Output: {:?}", dense_layer1.outputs);

    let mut activation_relu = ActivationReLU::new();
    activation_relu.forward(&dense_layer1.outputs);
    println!("ReLU Output: {:?}", activation_relu.outputs);

    let mut dense_layer2 = DenseLayer::new(3, 3);
    dense_layer2.forward(activation_relu.outputs);
    println!("Layer 2 Output: {:?}", dense_layer2.outputs);

    let mut activation_softmax = ActivationSoftMax::new();
    activation_softmax.forward(&dense_layer2.outputs);
    println!("SoftMax Output: {:?}", activation_softmax.outputs);
}
