# nnfs_r

An attempt to build a neural network from scratch based off the book [Neural Networks from Scratch in Python (Harrison Kinsley, Daniel Kukieła)](https://books.google.co.in/books/about/Neural_Networks_from_Scratch_in_Python.html?id=Ll1CzgEACAAJ&redir_esc=y) and brushing up Rust and ML concepts along with it.

## Project Structure

The project is organized into modular components for better maintainability and extensibility:

```
nnfs_r/
├── src/
│   ├── lib.rs                      # Library root with public exports
│   ├── main.rs                     # Demo application
│   ├── layers/                     # Neural network layers
│   │   ├── mod.rs                  # Layer trait definition
│   │   └── dense.rs                # Dense (fully connected) layer
│   ├── activations/                # Activation functions
│   │   ├── mod.rs                  # Activation trait
│   │   ├── relu.rs                 # ReLU activation
│   │   └── softmax.rs              # Softmax activation
│   ├── losses/                     # Loss functions
│   │   ├── mod.rs                  # Loss trait
│   │   ├── categorical_crossentropy.rs
│   │   └── combined.rs             # Combined softmax + cross-entropy
│   ├── optimizers/                 # Optimization algorithms
│   │   ├── mod.rs                  # Optimizer trait
│   │   └── sgd.rs                  # SGD with momentum and decay
│   └── utils.rs                    # Utility functions
├── examples/                       # Example usage
│   └── spiral_classification.rs
└── Cargo.toml
```

## Features

- **Modular Architecture**: Clean separation of concerns with trait-based design
- **Dense Layers**: Fully connected layers with forward and backward propagation
- **Activation Functions**: ReLU and Softmax implementations
- **Loss Functions**: Categorical cross-entropy with numerical stability
- **Optimizers**: SGD with momentum and learning rate decay
- **Comprehensive Tests**: Unit tests for all major components
- **Documentation**: Inline documentation with examples

## Usage

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
nnfs_r = { path = "../nnfs_r" }
```

Example usage:

```rust
use nnfs_r::{
    layers::DenseLayer,
    activations::ActivationReLU,
    optimizers::OptimizerSGD,
    utils::create_data,
};

fn main() {
    // Create dataset
    let (X, y) = create_data(100, 3);
    
    // Build network
    let mut layer1 = DenseLayer::new(2, 64);
    let mut activation = ActivationReLU::new();
    
    // Train...
}
```

### Running Examples

```bash
# Run the spiral classification example
cargo run --example spiral_classification

# Run the main demo
cargo run
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run tests for a specific module
cargo test layers::
```

## Development Principles

This project follows modern Rust best practices:

- **Trait-based design** for extensibility and polymorphism
- **Test-Driven Development** with comprehensive unit tests
- **Clear documentation** with examples for all public APIs
- **Type safety** leveraging Rust's strong type system
- **Performance** using efficient ndarray operations

## Future Enhancements

- [ ] Additional layer types (Dropout, BatchNorm, etc.)
- [ ] More activation functions (Sigmoid, Tanh, Leaky ReLU)
- [ ] Advanced optimizers (Adam, RMSprop, AdaGrad)
- [ ] Model serialization/deserialization
- [ ] GPU acceleration support
- [ ] Visualization tools for training progress
- [ ] More comprehensive examples and tutorials

## Contributing

Contributions are welcome! This project aims to be a learning resource for both Rust and neural networks. Please feel free to:

- Report bugs or suggest features via GitHub Issues
- Submit pull requests with improvements
- Add more examples or documentation
- Share your learning experience

## License

This project is open source and available for educational purposes.

## Acknowledgments

Based on the excellent book "Neural Networks from Scratch in Python" by Harrison Kinsley and Daniel Kukieła.
