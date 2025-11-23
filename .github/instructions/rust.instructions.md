---
applyTo: "**/*.rs"
---

# Rust Language Standards

## Overview

This file defines Rust-specific coding standards. These guidelines apply to all Rust code in the project and follow idiomatic Rust best practices.

## Code Style & Formatting

### General Formatting

- Follow `rustfmt.toml` configuration: max width 100, 4 spaces, Unix newlines
- Use field init shorthand and try shorthand (`?` operator)
- Force explicit ABI declarations
- Run `cargo fmt` before committing

### Naming Conventions

- **Variables**: Use `snake_case` (e.g., `learning_rate`, `weight_momentums`)
- **Functions/Methods**: Use `snake_case` (e.g., `forward`, `create_data`)
- **Types/Structs/Enums**: Use `PascalCase` (e.g., `DenseLayer`, `ActivationReLU`)
- **Constants**: Use `SCREAMING_SNAKE_CASE`
- **Trait names**: Use `PascalCase` (e.g., `Layer`, `Activation`)
- **Lifetimes**: Use short lowercase letters (e.g., `'a`, `'b`)
- **Type parameters**: Use single uppercase letter or `PascalCase` (e.g., `T`, `Item`)

### Module Organization

- One primary type per file (e.g., `dense.rs` for `DenseLayer`)
- Use `mod.rs` for trait definitions and re-exports
- Group related functionality in subdirectories: `layers/`, `activations/`, `losses/`, `optimizers/`
- Re-export commonly used items in `lib.rs` for convenience

## Documentation Standards

### Doc Comments

- **Required** for all public items (`///` for items, `//!` for modules)
- Include comprehensive module-level documentation explaining purpose
- Document all struct fields with inline comments
- Add `# Examples` section for public APIs
- Reference mathematical formulas where applicable (e.g., SGD update rule)
- Link to NNFS book concepts when relevant

### Documentation Structure

````rust
//! Brief module description
//!
//! Detailed explanation of the module's purpose and contents.
//!
//! # Features
//! - Feature 1
//! - Feature 2

/// Brief struct/function description
///
/// Detailed explanation of behavior and use cases.
///
/// # Arguments
/// * `param` - Description
///
/// # Returns
/// Description of return value
///
/// # Examples
/// ```rust
/// use nnfs_r::...;
/// // Example code
/// ```
````

## Clippy Configuration

### Enabled Lints

- `#![warn(missing_docs)]` - All public items must be documented
- `#![warn(clippy::all)]` - All standard clippy lints
- `#![warn(clippy::pedantic)]` - Pedantic lints for best practices

### Allowed Exceptions (with rationale)

- `clippy::module_name_repetitions` - Module names in type names can improve clarity
- `clippy::must_use_candidate` - Builder-style methods don't always need `#[must_use]`
- `clippy::missing_errors_doc` - Error types that are self-explanatory
- `clippy::missing_panics_doc` - Intentional panics in test code
- `clippy::cast_precision_loss` - Necessary for mathematical computations involving conversions

## Architecture Patterns

### Trait Design

- Define traits for common interfaces
- Use associated types when the type is determined by the implementation
- Use generic type parameters when the type is determined by the caller
- Provide default implementations for methods when sensible
- Keep traits focused and cohesive

### State Management

- Use private fields by default; expose through methods
- Make fields public only when direct access is justified
- Use `Cell` or `RefCell` for interior mutability when needed
- Prefer immutable data structures when possible
- Document ownership and borrowing patterns in complex cases

### Constructor Patterns

```rust
impl Type {
    /// Creates a new instance with required parameters
    pub fn new(required_param: usize) -> Self {
        Self {
            field: initialize_value(required_param),
            other_field: Default::default(),
        }
    }

    /// Creates a new instance with all defaults
    pub fn default() -> Self {
        Self::new(default_value)
    }
}

// Implement Default trait when appropriate
impl Default for Type {
    fn default() -> Self {
        Self::new(default_value)
    }
}
```

## Error Handling

### Panics

- Use `unwrap()` only in:
  - Test code
  - Examples where panic is acceptable
  - Cases where panic is truly unrecoverable
- Document panic conditions with `# Panics` section
- Prefer `expect("descriptive message")` over bare `unwrap()`

### Result Types

- Return `Result<T, E>` for operations that can fail
- Create custom error types for domain-specific errors
- Use the `?` operator to propagate errors
- Consider using `thiserror` or `anyhow` for error handling

Example:

```rust
/// # Errors
/// Returns error if the file cannot be read
pub fn load_data(path: &Path) -> Result<Data, LoadError> {
    let contents = std::fs::read_to_string(path)?;
    parse_data(&contents)
}
```

## Testing Requirements

### Test Organization

- Place tests in `#[cfg(test)] mod tests` at end of each file
- Use descriptive test function names that explain what is being tested
- Test both typical cases and edge cases
- Keep tests focused and independent

### Test Coverage

- Unit test all public methods
- Test error conditions and edge cases
- Verify correct behavior with different input types
- Use property-based testing for complex logic (consider `proptest` or `quickcheck`)

### Test Patterns

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_name() {
        let component = Component::new(params);
        let result = component.process(input);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_error_case() {
        let result = Component::from_invalid(data);
        assert!(result.is_err());
    }
}
```

## Performance Considerations

### General Optimization

- Prefer iterator chains over manual loops
- Use `collect()` judiciously; consider iterator consumers
- Avoid unnecessary allocations; use references and borrowing
- Use `clone_from()` instead of `clone()` when updating existing values
- Profile before optimizing; use `cargo bench` for benchmarking

### Memory Efficiency

- Pre-allocate collections with known sizes using `with_capacity()`
- Reuse buffers when processing multiple items
- Consider using `Cow` for copy-on-write semantics
- Use `Box`, `Rc`, or `Arc` for large data structures when appropriate

## Code Review Checklist

Before submitting code, verify:

- [ ] `cargo fmt` applied
- [ ] `cargo clippy` warnings addressed
- [ ] `cargo test` passes all tests
- [ ] All public items documented with examples
- [ ] No `TODO` or `FIXME` comments in final code
- [ ] Appropriate clippy allows with justification comments
- [ ] Tests added for new functionality
- [ ] Error handling is appropriate and documented
- [ ] No unnecessary allocations or clones
- [ ] Ownership and borrowing are clear and efficient

## Common Patterns to Follow

### Public Field Access

```rust
pub struct Type {
    /// Documentation for public field
    pub field: FieldType,
    // Private fields without pub
    internal: InternalType,
}
```

### Trait Implementation

```rust
impl TraitName for Type {
    fn method(&mut self, param: &ParamType) {
        // Implementation
    }

    fn accessor(&self) -> &FieldType {
        &self.field
    }
}
```

### Builder Pattern

```rust
pub struct Builder {
    field: Option<FieldType>,
}

impl Builder {
    pub fn new() -> Self {
        Self { field: None }
    }

    pub fn field(mut self, value: FieldType) -> Self {
        self.field = Some(value);
        self
    }

    pub fn build(self) -> Result<Type, BuildError> {
        Ok(Type {
            field: self.field.ok_or(BuildError::MissingField)?,
        })
    }
}
```

## Anti-Patterns to Avoid

- **Don't** use non-descriptive variable names (`a`, `b`, `tmp`) - use clear names
- **Don't** ignore clippy warnings without documented justification
- **Don't** create large monolithic files - split into logical modules
- **Don't** skip error handling - use `Result` for fallible operations
- **Don't** use `clone()` when borrowing would suffice
- **Don't** implement `Copy` for large structs - use references instead
- **Don't** use `unwrap()` in library code without documenting panic conditions
- **Don't** expose implementation details through public APIs
- **Don't** ignore compiler warnings - fix or explicitly allow with comment

## Best Practices

- **Prefer composition over inheritance** (Rust doesn't have inheritance)
- **Use the type system** to enforce invariants at compile time
- **Write self-documenting code** with clear names and structure
- **Keep functions small and focused** on a single responsibility
- **Use iterators** instead of index-based loops when possible
- **Leverage the borrow checker** rather than fighting it
- **Write tests** that document expected behavior
- **Consider API ergonomics** from the caller's perspective
