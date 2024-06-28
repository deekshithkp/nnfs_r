use ndarray::{s, Array, Array1, Array2, Axis, Zip};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;
use rand::Rng;
use std::f64::EPSILON;

#[derive(Debug)]
struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    outputs: Array2<f64>,
    dweights: Array2<f64>,
    dbiases: Array1<f64>,
    dinputs: Array2<f64>,
    weight_momentums: Array2<f64>,
    bias_momentums: Array1<f64>,
}

impl DenseLayer {
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

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.outputs = inputs.dot(&self.weights) + &self.biases;
    }

    pub fn backward(&mut self, inputs: &Array2<f64>, dvalues: &Array2<f64>) {
        self.dweights = inputs.t().dot(dvalues);
        self.dbiases = dvalues.sum_axis(Axis(0));
        self.dinputs = dvalues.dot(&self.weights.t());
    }
}

struct ActivationReLU {
    inputs: Array2<f64>,
    outputs: Array2<f64>,
}

impl ActivationReLU {
    pub fn new() -> Self {
        ActivationReLU {
            inputs: Array2::zeros((0, 0)),
            outputs: Array2::zeros((0, 0)),
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        self.inputs = inputs.clone();
        self.outputs = inputs.mapv(|x| x.max(0.0));
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
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

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        let max_values = inputs.map_axis(Axis(1), |row| *row.max().unwrap());
        let exp_values = (inputs - &max_values.insert_axis(Axis(1))).mapv(f64::exp);
        let sum_exp_values = exp_values.sum_axis(Axis(1));
        self.outputs = exp_values / &sum_exp_values.insert_axis(Axis(1));
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
        let mut dinputs = Array2::<f64>::zeros(dvalues.raw_dim());

        for (index, (single_output, single_dvalues)) in self.outputs.axis_iter(Axis(0)).zip(dvalues.axis_iter(Axis(0))).enumerate() {
            let single_output = single_output.into_shape((single_output.len(), 1)).unwrap();
            let jacobian_matrix = Array::eye(single_output.len()) - &single_output.dot(&single_output.t());
            let gradient = jacobian_matrix.dot(&single_dvalues);
            dinputs.slice_mut(s![index, ..]).assign(&gradient);
        }

        dinputs
    }
}

struct CategoricalCrossEntropyLoss;

impl CategoricalCrossEntropyLoss {
    pub fn new() -> Self {
        CategoricalCrossEntropyLoss {}
    }

    pub fn forward(y_pred: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
        let samples = y_pred.len_of(Axis(0));
        let clipped_values = y_pred.mapv(|x| x.max(EPSILON).min(1.0 - EPSILON));
        let correct_confidences = y_true.iter().enumerate().map(|(i, &label)| clipped_values[[i, label]]).collect::<Array1<f64>>();
        let negative_log_likelihoods = correct_confidences.mapv(|v| -v.ln());
        negative_log_likelihoods.sum() / samples as f64
    }

    pub fn backward(dvalues: &Array2<f64>, y_true: &Array1<usize>) -> Array2<f64> {
        let samples = dvalues.shape()[0];
        let labels = dvalues.shape()[1];

        let mut y_true_one_hot = Array2::<f64>::zeros((samples, labels));
        for (i, &label) in y_true.iter().enumerate() {
            y_true_one_hot[[i, label]] = 1.0;
        }

        let dinputs = -(&y_true_one_hot / dvalues) / samples as f64;
        dinputs
    }
}

struct ActivationSoftMaxLossCategoricalCrossEntropy {
    activation: ActivationSoftMax,
}

impl ActivationSoftMaxLossCategoricalCrossEntropy {
    pub fn new() -> Self {
        ActivationSoftMaxLossCategoricalCrossEntropy {
            activation: ActivationSoftMax::new(),
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>, y_true: &Array1<usize>) -> f64 {
        self.activation.forward(inputs);
        CategoricalCrossEntropyLoss::forward(&self.activation.outputs, y_true)
    }

    pub fn backward(&mut self, dvalues: &Array2<f64>, y_true: &Array1<usize>) -> Array2<f64> {
        let samples = dvalues.shape()[0];
        let classes = dvalues.shape()[1];

        let mut y_true_one_hot = Array2::<f64>::zeros((samples, classes));
        for (i, &label) in y_true.iter().enumerate() {
            y_true_one_hot[[i, label]] = 1.0;
        }

        let dinputs = (&self.activation.outputs - &y_true_one_hot) / samples as f64;
        dinputs
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
    pub fn new(learning_rate: f64, decay: f64, momentum: f64) -> Self {
        OptimizerSGD {
            learning_rate,
            current_learning_rate: learning_rate,
            decay,
            iteration: 0,
            momentum,
        }
    }

    pub fn pre_update_params(&mut self) {
        if self.decay != 0.0 {
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iteration as f64));
        }
    }

    pub fn update_params(&self, layer: &mut DenseLayer) {
        if self.momentum != 0.0 {
            let weight_updates = self.momentum * layer.weight_momentums.clone() - self.current_learning_rate * &layer.dweights;
            layer.weight_momentums = weight_updates.clone();

            let bias_updates = self.momentum * layer.bias_momentums.clone() - self.current_learning_rate * &layer.dbiases;
            layer.bias_momentums = bias_updates.clone();

            layer.weights += &weight_updates;
            layer.biases += &bias_updates;
        } else {
            layer.weights = layer.weights.clone() + self.current_learning_rate * &layer.dweights;
            layer.biases = layer.biases.clone() + self.current_learning_rate * &layer.dbiases;
        }
    }

    pub fn post_update_params(&mut self) {
        self.iteration += 1;
    }
}

fn create_data(samples: usize, classes: usize) -> (Array2<f64>, Array1<usize>) {
    let mut rng = rand::thread_rng();
    let mut X = Array2::<f64>::zeros((samples * classes, 2));
    let mut y = Array1::<usize>::zeros(samples * classes);

    for class_number in 0..classes {
        for sample in 0..samples {
            let ix = sample + class_number * samples;
            let r = rng.gen_range(0.0..1.0);
            let t = rng.gen_range(class_number as f64 * 4.0..(class_number + 1) as f64 * 4.0) + r * 0.2;
            X[[ix, 0]] = r * t.sin();
            X[[ix, 1]] = r * t.cos();
            y[ix] = class_number;
        }
    }

    (X, y)
}

fn accuracy(y_pred: &Array1<usize>, y_true: &Array1<usize>) -> f64 {
    y_pred.iter().zip(y_true.iter()).filter(|&(a, b)| a == b).count() as f64 / y_true.len() as f64
}

fn main() {
    let (X, y) = create_data(100, 3);

    let mut dense1 = DenseLayer::new(2, 64);
    let mut activation1 = ActivationReLU::new();
    let mut dense2 = DenseLayer::new(64, 3);
    let mut loss_activation = ActivationSoftMaxLossCategoricalCrossEntropy::new();
    let mut optimizer = OptimizerSGD::new(1.0, 1e-3, 0.9);

    let epochs = 10001;
    for epoch in 0..epochs {
        dense1.forward(&X);
        activation1.forward(&dense1.outputs);
        dense2.forward(&activation1.outputs);
        let data_loss = loss_activation.forward(&dense2.outputs, &y);
        let predictions = dense2.outputs.map_axis(Axis(1), |row| row.argmax().unwrap());
        let acc = accuracy(&predictions, &y);
        let loss = data_loss;

        if epoch % 1000 == 0 {
            println!("epoch: {}, acc: {:.3}, loss: {:.3}, lr: {:.3}", epoch, acc, loss, optimizer.current_learning_rate);
        }

        let dvalues = loss_activation.backward(&dense2.outputs, &y);
        dense2.backward(&activation1.outputs, &dvalues);
        let dvalues = activation1.backward(&dense2.dinputs);
        dense1.backward(&X, &dvalues);

        optimizer.pre_update_params();
        optimizer.update_params(&mut dense1);
        optimizer.update_params(&mut dense2);
        optimizer.post_update_params();
    }
}
