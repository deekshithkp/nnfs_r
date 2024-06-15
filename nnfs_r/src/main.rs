use ndarray::{Array, Array1, Array2, Axis, Ix1, s};
use ndarray_rand::{self, rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;
use rand::Rng;

#[derive(Debug)]
struct DenseLayer {
    inputs: Array2<f64>,
    weights: Array2<f64>,
    biases: Array1<f64>,
    outputs: Array2<f64>,

    dweights: Array2<f64>,
    dbiases: Array1<f64>,
    dinputs: Array2<f64>,

    weight_momentums: Option<Array2<f64>>,
    bias_momentums: Option<Array1<f64>>,
}

impl DenseLayer {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        DenseLayer {
            inputs: Array2::zeros((n_inputs, n_neurons)),
            weights: Array2::random((n_inputs, n_neurons), Uniform::new(0.0, 0.01)),
            biases: Array1::zeros((n_neurons)),
            outputs: Array2::zeros((1, n_neurons)),
            dweights: Array2::zeros((0, 0)),
            dbiases: Array1::zeros(0),
            dinputs: Array2::zeros((0, 0)),
            weight_momentums: None,
            bias_momentums: None,
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = inputs.clone();
        self.outputs = inputs.dot(&self.weights) + &self.biases;
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        // Gradients on parameter
        self.dweights = self.inputs.t().dot(dvalues);
        self.dbiases = dvalues.sum_axis(Axis(0));
        self.dinputs = dvalues.dot(&self.weights.t());
    }
}

struct ActivationReLU {
    inputs: Array2<f64>,
    outputs: Array2<f64>,
    dinputs: Array2<f64>,
}

impl ActivationReLU {
    pub fn new() -> Self {
        ActivationReLU {
            inputs: Array2::zeros((0, 0)),
            dinputs: Array2::zeros((0, 0)),
            outputs: Array2::zeros((0, 0)),
        }
    }

    // ReLU is a linear function that outputs the input for all + inputs and 0 for -ve inputs
    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = inputs.clone();
        self.outputs = inputs.mapv(|i| i.max(0.0));
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) {
        self.dinputs = dvalues.clone();
        
        // Zero gradient where input values were negative
        let inputs = &self.inputs;
        for ((i, j), value) in self.dinputs.indexed_iter_mut() {
            if inputs[[i, j]] <= 0.0 {
                *value = 0.0;
            }
        }
    }
}

struct ActivationSoftMax {
    dinputs: Array2<f64>,
    outputs: Array2<f64>,
}

impl ActivationSoftMax {
    pub fn new() -> Self {
        ActivationSoftMax {
            dinputs: Array2::zeros((0, 0)),
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

    fn backward(&mut self, dvalues: &Array2<f64>) {
        // Create uninitialized array
        let mut dinputs = Array::zeros(dvalues.raw_dim());

        // Enumerate outputs and gradients
        for (index, (single_output, single_dvalues)) in self.outputs.axis_iter(Axis(0)).zip(dvalues.axis_iter(Axis(0))).enumerate() {
            // Flatten output array
        
            let single_output = single_output.into_shape((single_output.len(), 1)).unwrap();
            
            // Calculate Jacobian matrix of the output
            let jacobian_matrix = Array::<f64, _>::eye(single_output.len()) - &single_output.dot(&single_output.t());

            // Calculate sample-wise gradient
            let gradient = jacobian_matrix.dot(&single_dvalues);

            // Add it to the array of sample gradients
            dinputs.slice_mut(s![index, ..]).assign(&gradient);
        }

        self.dinputs = dinputs;
    }

}

struct CategoricalCrossEntropyLoss {
    output: f64,
    dinputs: Array2<f64>,
}

impl CategoricalCrossEntropyLoss {
    fn new() -> Self {
        CategoricalCrossEntropyLoss {
            output: 0.,
            dinputs: Array2::zeros((0, 0)),
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

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<f64>) {
        // Number of samples
        let samples = dvalues.shape()[0];

        // Number of labels in each sample - we use the first sample to count them
        let labels = dvalues.shape()[1];

        // Convert sparse labels to one-hot vector if necessary
        let y_true_one_hot = if y_true.shape().len() == 1 {
            let mut one_hot = Array::zeros((samples, labels));
            for (i, &label) in y_true.iter().enumerate() {
                one_hot[(i, label as usize)] = 1.0;
            }
            one_hot
        } else {
            y_true.clone()
        };
        
        // Calculate gradient
        let mut dinputs = -&y_true_one_hot / dvalues;
        // Normalize gradient
        dinputs /= samples as f64;

        self.dinputs =dinputs;

    }
}


struct ActivationSoftMaxLossCategoricalCrossEntropy {
    activation: ActivationSoftMax,
    loss: CategoricalCrossEntropyLoss,
    output: Array2<f64>,
    dinputs: Array2<f64>,
}

impl ActivationSoftMaxLossCategoricalCrossEntropy {
    fn new() -> Self {
        ActivationSoftMaxLossCategoricalCrossEntropy {
            activation: ActivationSoftMax::new(),
            loss: CategoricalCrossEntropyLoss::new(),
            output: Array2::zeros((0, 0)),
            dinputs: Array2::zeros((0, 0)),
        }
    }

    fn forward(&mut self, inputs: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        self.activation.forward(inputs);
        self.output = self.activation.outputs.clone();
        self.loss.calculate(&self.output, y_true);
        return self.loss.output;
    }

    fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array2<f64>) {
        let samples = dvalues.shape()[0];

        let y_true =y_true.map_axis(Axis(1), |row| row.argmax().unwrap());

        self.dinputs = dvalues.clone();
        for i in 0..self.dinputs.nrows() {
            self.dinputs[(i, y_true[i])] -= 1.0;
        }

        self.dinputs /= samples as f64;

    }
}

struct OptimizerSGD {
    learning_rate: f64,
    current_learning_rate: f64,
    decay: f64,
    iteration: usize,
    momentum: f64,
}

impl OptimizerSGD {
    fn new(learning_rate: f64, decay: f64, momentum: f64) -> Self {
        OptimizerSGD {
            learning_rate,
            current_learning_rate: learning_rate,
            decay,
            iteration: 0,
            momentum: momentum,
        }
    }

    fn pre_update_params(&mut self) {
        if self.decay != 0. {
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration as f64));
        }
    }

    fn update_params(&self, layer: &mut DenseLayer) {

        if self.momentum != 0. {
            if layer.weight_momentums.is_none() {
                layer.weight_momentums = Some(Array2::zeros(layer.weights.raw_dim()));
                layer.bias_momentums = Some(Array1::zeros(layer.biases.raw_dim()));
            }

            let weight_updates = self.momentum * layer.weight_momentums.as_ref().unwrap()
                - self.current_learning_rate * &layer.dweights;
            layer.weight_momentums = Some(weight_updates.clone());

            let bias_updates = self.momentum * layer.bias_momentums.as_ref().unwrap()
                - self.current_learning_rate * &layer.dbiases;
            layer.bias_momentums = Some(bias_updates.clone());

            layer.weights = &layer.weights + weight_updates.clone();
            layer.biases = &layer.biases + bias_updates.clone();
        }
        else {
            let weight_updates = self.current_learning_rate * &layer.dweights;
            let bias_updates = self.current_learning_rate * &layer.dbiases;

            layer.weights = &layer.weights + weight_updates;
            layer.biases = &layer.biases + bias_updates;
        }
    }

    fn post_update_params(&mut self) {
        self.iteration += 1;
    }
}

fn main() {
    // Notice that the shape for the inputs and Neuron's weights match
    // Random data is fine for now but we would need to have the sample data in a persistent storage (likely a file) when we get to training the network
    let (inputs, y) = create_data(100, 3);
    let mut dense_layer1 = DenseLayer::new(2, 3);
    let mut activation_relu = ActivationReLU::new();

    let mut dense_layer2 = DenseLayer::new(3, 3);
    let mut loss_activation = ActivationSoftMaxLossCategoricalCrossEntropy::new();

    let mut optimiser = OptimizerSGD::new(1., 1e-3, 0.9);

    for epoch in 0..10001 {
        dense_layer1.forward(&inputs);
        activation_relu.forward(&dense_layer1.outputs);
        dense_layer2.forward(&activation_relu.outputs);
    
        let loss = loss_activation.forward(&dense_layer2.outputs, &y);   
    
        let predictions = loss_activation.output
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
    
        if epoch % 100 == 0 {
            println!("Epoch: {epoch}\tAccuracy: {accuracy}\tLoss: {loss}\tLearning rate: {}",optimiser.current_learning_rate);
        }

            // Backward pass
        loss_activation.backward(&loss_activation.output.clone(), &y);
        dense_layer2.backward(&loss_activation.dinputs);
        activation_relu.backward(&dense_layer2.dinputs);
        dense_layer1.backward(&activation_relu.dinputs);

        optimiser.pre_update_params();
        optimiser.update_params(&mut dense_layer1);
        optimiser.update_params(&mut dense_layer2);
        optimiser.post_update_params();
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