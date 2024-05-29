use ndarray::{Array1, Array2, ArrayBase, Axis, Ix1};
use ndarray_rand::{self, rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;
use rand::Rng;

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

    pub fn forward(&mut self, inputs: &Array2<f64>) {
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

    fn forward(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> Array1<f64> {
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

    fn calculate(&mut self, output: &Array2<f64>, y: &Array2<f64>) {
        let sample_losses = Self::forward(output, y);

        // Calculate mean loss
        let data_loss = sample_losses.mean();
        self.output = data_loss.unwrap();
    }
}

fn main() {
    // Notice that the shape for the inputs and Neuron's weights match
    // Random data is fine for now but we would need to have the sample data in a persistent storage (likely a file) when we get to training the network
    let (inputs, y) = create_data(100, 3);
    let mut dense_layer1 = DenseLayer::new(2, 3);
    let mut activation_relu = ActivationReLU::new();

    let mut dense_layer2 = DenseLayer::new(3, 3);
    let mut activation_softmax = ActivationSoftMax::new();

    let mut loss_function = CategoricalCrossEntropyLoss::new();

    // Helper variables
    let mut lowest_loss = 9999999.; // some initial value
    let mut best_dense1_weights = dense_layer1.weights.clone();
    let mut best_dense1_biases = dense_layer1.biases.clone();
    let mut best_dense2_weights = dense_layer2.weights.clone();
    let mut best_dense2_biases = dense_layer2.biases.clone();

    for iteration in 0..=10000 {
        // tweak weights and biases with small values in the hope of improving accuracy
        dense_layer1.weights = dense_layer1.weights.mapv(|v| v + (0.05 * rand::random::<f64>()));
        dense_layer1.biases = dense_layer1.biases.mapv(|v| v + (0.05 * rand::random::<f64>()));
        dense_layer2.weights = dense_layer2.weights.mapv(|v| v + (0.05 * rand::random::<f64>()));
        dense_layer2.biases = dense_layer2.biases.mapv(|v| v + (0.05 * rand::random::<f64>()));

        // perform a forward pass 
        dense_layer1.forward(&inputs);
        activation_relu.forward(&dense_layer1.outputs);
        dense_layer2.forward(&activation_relu.outputs);
        activation_softmax.forward(&dense_layer2.outputs);

        // calculate the loss
        loss_function.calculate(&activation_softmax.outputs, &y);
        let loss = loss_function.output;


        // calculate accuracy from softmax outputs against y
        let predictions = activation_softmax.outputs
            .map_axis(Axis(1), |row| row.argmax().unwrap())
            .into_dimensionality::<Ix1>()
            .unwrap();
        let y_for_acc = y
            .map_axis(Axis(1), |row| row.argmax().unwrap())
            .into_dimensionality::<Ix1>()
            .unwrap();

        let accuracy = predictions
            .iter()
            .zip(y_for_acc.iter())
            .map(|(p, t)| if p == t { 1.0 } else { 0.0 })
            .sum::<f64>()
            / predictions.len() as f64;


        // if loss is smaller, print and save weights & biases
        if loss < lowest_loss {
            println!("New set of weights found: Iteration - {iteration}, Loss: {loss}, Accuracy: {accuracy} ");
            best_dense1_weights = dense_layer1.weights.clone();
            best_dense1_biases = dense_layer1.biases.clone();
            best_dense2_weights = dense_layer2.weights.clone();
            best_dense2_biases = dense_layer2.biases.clone();

            lowest_loss = loss;
        } 
        // revert previous values if not improved
        else {
            dense_layer1.weights = best_dense1_weights.clone();
            dense_layer1.biases = best_dense1_biases.clone();
            dense_layer2.weights = best_dense2_weights.clone();
            dense_layer2.biases = best_dense2_biases.clone();
        }
    }   


} 


fn create_data(samples: usize, classes: usize) -> (Array2<f64>, Array2<f64>) {
    let mut rng = rand::thread_rng();
    let mut X = Array2::zeros((samples * classes, 2));
    let mut y = Array2::zeros((samples * classes, classes));

    for class_number in 0..classes {
        let start = samples * class_number;
        let end = samples * (class_number + 1);
        let r = ndarray::Array::linspace(0.0, 1.0, samples);
        let mut t = ndarray::Array::linspace(class_number as f64 * 4.0, (class_number + 1) as f64 * 4.0, samples);
        t = t + rng.gen_range(-0.1..0.1);

        for i in start..end {
            let x = r[i - start] * t[i - start] * 2.5;
            X[[i, 0]] = x.sin();
            X[[i, 1]] = x.cos();
            y[[i, class_number]] = 1.0;
        }
    }

    (X, y)
}