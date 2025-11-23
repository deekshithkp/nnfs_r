---
applyTo: "**/*"
---

# nnfs_r Domain-Specific Guidelines

## Project Context

This is a neural network library built from scratch in Rust, following "Neural Networks from Scratch in Python" by Harrison Kinsley and Daniel Kukieła. The codebase emphasizes educational clarity, mathematical correctness, and practical implementation of ML concepts.

## Project Philosophy

### Educational Code Principles

- **Clarity over cleverness**: Prefer readable code to overly terse implementations
- **Comments for math**: Explain non-obvious mathematical operations with formulas
- **Examples matter**: Every public API should have working examples
- **Test as documentation**: Tests demonstrate intended usage patterns
- **Performance with understanding**: Optimize but explain why

### Learning Goals

- Understand neural network internals by building from scratch
- Bridge theory (NNFS book) with practice (Rust implementation)
- Balance educational clarity with production-quality code
- Document the "why" behind implementation choices

## Neural Network Architecture

### Component Organization

```
src/
├── layers/         # Neural network layers (Dense, etc.)
├── activations/    # Activation functions (ReLU, Softmax, etc.)
├── losses/         # Loss functions (CrossEntropy, etc.)
├── optimizers/     # Optimization algorithms (SGD, Adam, etc.)
└── utils.rs        # Dataset generation, metrics
```

### Trait-Based Design

Each component type has a corresponding trait:

- **`Layer`**: Forward/backward propagation interface
- **`Activation`**: Activation function interface
- **`Loss`**: Loss computation and gradient interface
- **`Optimizer`**: Parameter update interface

### Forward/Backward Pass Pattern

All components follow this pattern:

```rust
fn forward(&mut self, inputs: &Array2<f64>) {
    self.inputs.clone_from(inputs); // Store for backward pass
    self.outputs = compute_outputs(inputs);
}

fn backward(&mut self, dvalues: &Array2<f64>) -> Array2<f64> {
    // Compute gradients using stored inputs
    let dinputs = compute_dinputs(&self.inputs, dvalues);
    dinputs
}
```

**Key principles:**

- Store inputs during forward pass for use in backward pass
- Return gradients with respect to inputs
- Maintain consistent tensor shapes throughout

### State Management

Components must store:

- **Inputs**: Original inputs from forward pass (for gradient computation)
- **Outputs**: Results of forward pass (for next layer)
- **Gradients**: `dweights`, `dbiases`, `dinputs` (for optimization)
- **Optimizer state**: Momentum, cache values (for adaptive learning rates)

## Machine Learning Conventions

### Naming Conventions

- **`X`**: Feature matrix (inputs), shape `(n_samples, n_features)`
- **`y`**: Target labels, shape `(n_samples,)` or `(n_samples, n_classes)`
- **`dvalues`**: Gradient flowing backward through the network
- **`dinputs`**: Gradient with respect to layer inputs

Use `#[allow(non_snake_case)]` for functions using ML conventions.

### Tensor Shapes

Always maintain clear shape expectations:

- **Batch dimension first**: `(batch_size, features)`
- **Weight matrices**: `(input_features, output_features)`
- **Biases**: `(output_features,)` - broadcasted across batch

Document expected shapes in function comments:

```rust
/// # Arguments
/// * `inputs` - Input data, shape (batch_size, n_features)
///
/// # Output
/// Shape (batch_size, n_neurons)
```

## Numerical Stability

### Critical Considerations

1. **Prevent division by zero**: Add small epsilon (`1e-7`) to denominators
2. **Clip gradients**: Prevent overflow with `clamp(-1.0, 1.0)` or similar
3. **Stable softmax**: Subtract max before exponential to prevent overflow
4. **Log operations**: Use `max(value, 1e-7).ln()` to prevent `-inf`

### Example: Stable Softmax

```rust
// Subtract max for numerical stability
let max_values = inputs.map_axis(Axis(1), |row| *row.max().unwrap());
let exp_values = (inputs - &max_values.insert_axis(Axis(1))).mapv(f64::exp);
```

### Testing Numerical Stability

Include tests with edge cases:

- Very small values (near zero)
- Very large values (potential overflow)
- Negative values
- Zero inputs

## Weight Initialization

### Standard Approaches

1. **Small random values**: `Uniform::new(-0.01, 0.01)` for simple cases
2. **Xavier/Glorot**: For networks with sigmoid/tanh activations
3. **He initialization**: For networks with ReLU activations
4. **Zero biases**: Standard practice across all layers

Document the initialization strategy:

```rust
/// Weights are initialized using a uniform distribution in the range [-0.01, 0.01]
/// to break symmetry while keeping values small.
```

## Gradient Computation

### Backpropagation Rules

Each component must compute three gradients:

1. **`dweights`**: Gradient with respect to weights
   - Formula: `inputs.T @ dvalues`
2. **`dbiases`**: Gradient with respect to biases
   - Formula: `sum(dvalues, axis=0)`
3. **`dinputs`**: Gradient with respect to inputs (for previous layer)
   - Formula: `dvalues @ weights.T`

### Gradient Checking

- Verify gradient shapes match parameter shapes
- Test gradients with known analytical solutions when possible
- Use gradient checking (numerical gradients) for complex operations

## Loss Functions

### Requirements

- Implement forward pass (compute loss value)
- Implement backward pass (compute gradient)
- Handle both single samples and batches
- Support regularization terms when applicable

### Combined Operations

For efficiency, combine operations where mathematically beneficial:

- `ActivationSoftMaxLossCategoricalCrossEntropy`: Combines softmax + cross-entropy
- Simplified gradient: `predictions - targets`

## Optimizers

### Standard Interface

All optimizers implement:

- `pre_update_params()`: Update learning rate (decay, etc.)
- `update_params()`: Update layer parameters
- `post_update_params()`: Increment iteration counter

### Momentum and Adaptive Learning

- Initialize momentum arrays to zero
- Update momentum before applying to parameters
- Document momentum formulas in comments

## Dataset Utilities

### Data Generation

- Provide toy datasets for quick experimentation (spiral, circles, etc.)
- Return tuple `(X, y)` with clear shapes
- Use reproducible random seeds in examples

### Metrics

- `accuracy()`: Classification accuracy
- Future: precision, recall, F1-score, confusion matrix

## Testing Strategy

### Unit Tests

Test each component in isolation:

- Forward pass produces correct shapes
- Backward pass produces correct gradient shapes
- Known inputs produce expected outputs
- Edge cases don't crash or produce NaN/Inf

### Integration Tests

Place in `examples/`:

- Full training loops (e.g., `spiral_classification.rs`)
- Demonstrate convergence on toy datasets
- Show typical usage patterns

### Test Data

Use small, controlled datasets:

```rust
let inputs = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
let expected_shape = (2, 3); // batch_size=2, neurons=3
```

## Documentation Requirements

### Mathematical Context

Document formulas and algorithms:

```rust
/// # Formula
///
/// Without momentum:
/// - weight = weight - learning_rate * gradient
///
/// With momentum:
/// - velocity = momentum * velocity - learning_rate * gradient
/// - weight = weight + velocity
```

### NNFS Book References

Link to relevant chapters when applicable:

```rust
/// Based on Chapter 5 of "Neural Networks from Scratch in Python"
```

### Examples in Docs

Every public API should include usage example:

````rust
/// # Examples
///
/// ```rust
/// use nnfs_r::layers::DenseLayer;
///
/// let mut layer = DenseLayer::new(2, 64);
/// // ... use layer
/// ```
````

## Performance Considerations

### `ndarray` Best Practices

- Use `dot()` for matrix multiplication (BLAS-accelerated)
- Use `mapv()` for element-wise operations
- Use `Zip::from()` for multi-array operations
- Avoid unnecessary allocations in hot paths

### Memory Management

- Pre-allocate output arrays in constructors
- Reuse buffers in forward/backward passes
- Use `clone_from()` instead of `clone()` when updating stored state

### Optimization Priorities

1. **Correctness first**: Get the math right
2. **Clarity second**: Make it understandable
3. **Performance third**: Optimize hot paths with profiling

## Common Pitfalls

### Avoid These Mistakes

- **Shape mismatches**: Always verify tensor dimensions
- **Forgotten gradient storage**: Store inputs during forward pass
- **Unstable operations**: Add epsilon to divisions and logarithms
- **Mixed conventions**: Don't alternate between `X`/`y` and `inputs`/`labels`
- **Missing initialization**: Initialize all arrays (especially momentum)
- **Batch dimension errors**: Remember first dimension is always batch
