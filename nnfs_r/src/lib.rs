//! Neural Networks from Scratch in Rust
//!
//! A comprehensive neural network library built from scratch, following the book
//! "Neural Networks from Scratch in Python" by Harrison Kinsley and Daniel Kukieła.
//!
//! # Features
//!
//! - Modular architecture with clean separation of concerns
//! - Dense (fully connected) layers
//! - Multiple activation functions (ReLU, Softmax)
//! - Loss functions with gradient computation
//! - SGD optimizer with momentum and learning rate decay
//! - Example datasets for quick experimentation
//!
//! # Example
//!
//! ```rust,no_run
//! use nnfs_r::layers::DenseLayer;
//! use nnfs_r::activations::ActivationReLU;
//! use nnfs_r::optimizers::OptimizerSGD;
//!
//! // Create a simple network
//! let mut layer = DenseLayer::new(2, 64);
//! let mut activation = ActivationReLU::new();
//! let mut optimizer = OptimizerSGD::new(0.1, 0.0, 0.0);
//! ```

pub mod activations;
pub mod layers;
pub mod losses;
pub mod optimizers;
pub mod utils;

// Re-export commonly used items for convenience
pub use activations::{ActivationReLU, ActivationSoftMax};
pub use layers::DenseLayer;
pub use losses::{ActivationSoftMaxLossCategoricalCrossEntropy, CategoricalCrossEntropyLoss};
pub use optimizers::OptimizerSGD;
pub use utils::{accuracy, create_data};
