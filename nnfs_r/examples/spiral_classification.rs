//! Spiral Classification Example
//!
//! This example demonstrates training a simple neural network on a spiral dataset.
//! It uses a two-layer network with ReLU activation and softmax output for
//! multi-class classification.
//!
//! Run this example with:
//! ```bash
//! cargo run --example spiral_classification
//! ```

use ndarray::Axis;
use ndarray_stats::QuantileExt;
use nnfs_r::{
    activations::{Activation, ActivationReLU},
    layers::{DenseLayer, Layer},
    losses::ActivationSoftMaxLossCategoricalCrossEntropy,
    optimizers::{Optimizer, OptimizerSGD},
    utils::{accuracy, create_data},
};

fn main() {
    // Generate spiral dataset: 100 samples per class, 3 classes
    let (X, y) = create_data(100, 3);

    // Initialize network layers
    let mut dense1 = DenseLayer::new(2, 64);
    let mut activation1 = ActivationReLU::new();
    let mut dense2 = DenseLayer::new(64, 3);
    let mut loss_activation = ActivationSoftMaxLossCategoricalCrossEntropy::new();

    // Initialize optimizer with learning rate, decay, and momentum
    let mut optimizer = OptimizerSGD::new(1.0, 1e-3, 0.9);

    let epochs = 10001;
    println!("Training neural network on spiral dataset...");
    println!("Network architecture: 2 -> 64 (ReLU) -> 3 (Softmax)\n");

    for epoch in 0..epochs {
        // Forward pass
        dense1.forward(&X);
        activation1.forward(dense1.outputs());
        dense2.forward(activation1.outputs());
        let data_loss = loss_activation.forward(dense2.outputs(), &y);

        // Calculate accuracy
        let predictions = dense2
            .outputs()
            .map_axis(Axis(1), |row| row.argmax().unwrap());
        let acc = accuracy(&predictions, &y);

        // Print progress every 1000 epochs
        if epoch % 1000 == 0 {
            println!(
                "Epoch: {:5} | Accuracy: {:.3} | Loss: {:.3} | LR: {:.4}",
                epoch,
                acc,
                data_loss,
                optimizer.current_learning_rate()
            );
        }

        // Backward pass
        let dvalues = loss_activation.backward(dense2.outputs(), &y);
        dense2.backward(activation1.outputs(), &dvalues);
        let dvalues = activation1.backward(dense2.dinputs());
        dense1.backward(&X, &dvalues);

        // Update parameters
        optimizer.pre_update_params();
        optimizer.update_params(&mut dense1);
        optimizer.update_params(&mut dense2);
        optimizer.post_update_params();
    }

    println!("\nTraining completed!");
}
