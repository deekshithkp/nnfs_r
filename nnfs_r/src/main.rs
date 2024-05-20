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

    // notice the change in the dot product calculation (this is the not the ideal way to support the array shape)
    let layer_outputs = weights.dot(&inputs) + biases;

    // Layer of Neurons with 3 nuerons with a batch of inputs (sample)
    let inputs = arr2(&[
            [1.0, 2.0, 3.0, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
    ]);
    let weights = arr2(&[
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
        ]);

    let biases = arr1(&[2.0, 3.0, 0.5]);

    // Matrix multiplication is employed here since we have a batch of inputs
    // a.b = abT
    
    // Notice that we have again reversed the parameters back to the inputs as first parameter for the dot product
    // This is because it would be ideal to have the output layer be defined by the input sample rather than the no. of neurons / their weights
    // Refer page 37
    let layer_outputs = inputs.dot(&weights.t()) + biases;

    println!("Neuron Layer output: {:?}", layer_outputs);

    // Dense layer concepts with 4 inputs, and two hidden layers of 3 neurons each
    let inputs = arr2(&[
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]);
    let weights = arr2(&[
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]);

    let biases = arr1(&[2.0, 3.0, 0.5]);

    let weights2 = arr2(&[
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]]);
    let biases2 = arr1(&[-1.0, 2.0, -0.5]);

    let layer1_outputs = inputs.dot(&weights.t()) + biases;
    let dense_layer_outputs = layer1_outputs.dot(&weights2.t()) + biases2;

    println!("Neuron Layer output: {:?}", dense_layer_outputs);
}
