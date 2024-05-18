use std::iter::zip;


fn main() {
    // Single Neuron concepts with 3 inputs (each with a bias)
    // Each neuron holds a weight
    // Concepts for introduction purposes; would be scratched off later

    let inputs = [1.0, 2.0, 3.0, 2.5];
    let weights = [0.2, 0.8, -0.5, 1.0];
    let bias = 2.0;

    let dot_product: f64 = inputs.iter()
        .zip(weights.iter())
        .map(|(input, weight)| input * weight)
        .sum();

    let output = dot_product + bias;

    println!("Neuron output: {output}");

    // Layer of Neurons concepts with 3 neurons connected to the same input but with different weights and biases
    let inputs = [1.0, 2.0, 3.0, 2.5];
    let weights = [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87],
        ];
    let biases = [2.0, 3.0, 0.5];

    let mut layer_outputs: Vec<f64> = Vec::new();

    for (neuron_weights, neuron_bias) in zip(weights, biases){
        let mut neuron_output = 0.0;

        for (n_input, weight) in zip(inputs, neuron_weights) {
            neuron_output += n_input * weight;
        }

        neuron_output += neuron_bias;

        layer_outputs.push(neuron_output);
    }

    println!("Neuron Layer output: {:?}", layer_outputs);
}
