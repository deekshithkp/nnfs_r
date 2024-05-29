use ndarray::{Axis,Array1,Array2};
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
            outputs: Array2::zeros((1, n_neurons)),
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

struct CategoricalCrossEntropyLoss {
    output: f64,
}

impl CategoricalCrossEntropyLoss {
    fn new() -> Self {
        CategoricalCrossEntropyLoss {
            output: 0.,
        }
    }

    fn forward(y_pred: Array2<f64>, y_true: Array2<f64>) -> Array1<f64> {
        // Number of samples in a batch
        let samples = y_pred.len_of(Axis(0));

        // Clip data to a non-zero number (insignificant magnitude) to avoid divide-by-zero hell
        // Both ends are clipped so as to not influence the mean value
        let clipped_values = y_pred.mapv(|x| x.max(1e-7).min(1.0 - 1e-7));

        // Probabilities for target values - account for both class labels as well as one-hot encoded labels
        let correct_confidences = 
        if y_true.shape().len() ==1 {
            // class labels
            let mut confidences = Array1::<f64>::zeros(samples);

            for i in 0..samples {
                confidences[i] = clipped_values[[i, y_true[[i, 0]] as usize]];
            }
            confidences
        } else {
            // one-hot code encoded labels
            (clipped_values * y_true).sum_axis(Axis(1))
        };

        // Losses
        let negative_likelyhoods = correct_confidences.mapv(|v| -v.ln());
        negative_likelyhoods

    }

    fn calculate(&mut self, output: Array2<f64>, y: Array2<f64>) {
        let sample_losses = Self::forward(output, y);

        // Calculate mean loss
        let data_loss = sample_losses.mean();
        self.output = data_loss.unwrap();
    }
}

fn main() {
    // Notice that the shape for the inputs and Neuron's weights match
    // Random data is fine for now but we would need to have the sample data in a persistent storage (likely a file) when we get to training the network
    let inputs = Array2::random((6, 4), Uniform::new(-2.5, 2.53));
    let y = Array2::random((6, 3), Uniform::new(0., 2.46));
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

    let mut loss = CategoricalCrossEntropyLoss::new();
    loss.calculate(activation_softmax.outputs, y);
    println!("Loss: {}", loss.output);
}
