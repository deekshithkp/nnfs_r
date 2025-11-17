# Contributing to nnfs_r

Thank you for your interest in contributing to nnfs_r! This project is both a learning resource and a practical implementation of neural networks from scratch.

## Ways to Contribute

- **Report Bugs**: Open an issue describing the bug and how to reproduce it
- **Suggest Enhancements**: Propose new features or improvements
- **Improve Documentation**: Fix typos, add examples, or clarify explanations
- **Add Tests**: Increase test coverage or add edge case tests
- **Implement Features**: Add new layer types, activations, optimizers, etc.
- **Share Examples**: Create example applications demonstrating the library

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/deekshithkp/nnfs_r.git
   cd nnfs_r/nnfs_r
   ```

2. **Build the project**
   ```bash
   cargo build
   ```

3. **Run tests**
   ```bash
   cargo test
   ```

4. **Run examples**
   ```bash
   cargo run --example spiral_classification
   ```

## Coding Guidelines

### Rust Best Practices

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for code formatting: `cargo fmt`
- Run `clippy` for linting: `cargo clippy`
- Ensure all tests pass: `cargo test`

### Code Style

- Use clear and descriptive variable and function names
- Adhere to `.editorconfig` settings
- Follow consistent indentation and spacing
- Write modular and reusable code
- Use modern Rust features where appropriate
- Consider performance optimizations
- Ensure code is well-documented with comments where necessary
- Follow SOLID principles and design patterns
- Write testable code
- Avoid code duplication

### Documentation

- Add doc comments (`///`) for all public items
- Include examples in doc comments where helpful
- Explain complex algorithms or mathematical operations
- Reference the NNFS book where relevant
- Update README.md if adding new features

### Testing

- Write unit tests for new functionality
- Ensure tests are deterministic and reproducible
- Test edge cases and error conditions
- Use descriptive test names that explain what is being tested
- Aim for high test coverage

## Project Structure

The project follows a modular architecture:

- **`src/layers/`**: Neural network layer implementations
- **`src/activations/`**: Activation function implementations
- **`src/losses/`**: Loss function implementations
- **`src/optimizers/`**: Optimization algorithm implementations
- **`src/utils.rs`**: Utility functions
- **`examples/`**: Example applications
- **`tests/`**: Integration tests (if needed)

## Pull Request Process

1. **Create a branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding guidelines

3. **Add tests** for new functionality

4. **Update documentation** as needed

5. **Run tests and linting**
   ```bash
   cargo fmt
   cargo clippy
   cargo test
   ```

6. **Commit your changes** with clear commit messages
   ```bash
   git commit -m "Add feature: description"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request** with a clear description of your changes

## Adding New Components

### Adding a New Layer Type

1. Create a new file in `src/layers/`
2. Implement the `Layer` trait
3. Add comprehensive tests
4. Export in `src/layers/mod.rs`
5. Add example usage in documentation
6. Update README.md if it's a major addition

### Adding a New Activation Function

1. Create a new file in `src/activations/`
2. Implement the `Activation` trait
3. Add tests for forward and backward passes
4. Export in `src/activations/mod.rs`
5. Document mathematical properties

### Adding a New Optimizer

1. Create a new file in `src/optimizers/`
2. Implement the `Optimizer` trait
3. Add tests for parameter updates
4. Document the algorithm and hyperparameters
5. Export in `src/optimizers/mod.rs`

## Questions?

Feel free to open an issue for any questions about contributing!

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on learning and improvement
- Help others learn from your contributions

Thank you for contributing to nnfs_r! 🦀🧠
