use std::iter::zip;
use ndarray::{arr1, arr2};


fn main() {
    // Single Neuron concepts with 3 inputs (each with a bias)
    // Each neuron holds a weight
    // Implementation changed to use ndarray - dot product

    let inputs = arr1(&[1.0, 2.0, 3.0, 2.5]);
    let weights = arr1(&[0.2, 0.8, -0.5, 1.0]);
    let bias = 2.0;

    let output: f64 = inputs.dot(&weights) + bias;
    println!("Neuron output: {output}");

    // Layer of Neurons concepts with 3 neurons connected to the same input but with different weights and biases
    // Implementation changed to use ndarray - dot product
    let inputs = arr1(&[1.0, 2.0, 3.0, 2.5]);
    let weights = arr2(&[
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87],
        ]);
    let biases = arr1(&[2.0, 3.0, 0.5]);

    // notice the change in the dot product calculation (this is the correct way to support the array shape)
    let layer_outputs = weights.dot(&inputs) + biases;

    println!("Neuron Layer output: {:?}", layer_outputs);
}
