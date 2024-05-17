
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

    print!("Neuron output: {output}");
}
