# Accessing Candle from Python with PyO3

## Introduction

While Rust offers exceptional performance and safety guarantees that make Candle a powerful deep learning framework, Python remains the dominant language in the machine learning ecosystem. Many data scientists and machine learning practitioners are more comfortable with Python's syntax and have existing Python-based workflows. Additionally, the Python ecosystem includes popular libraries like NumPy, Pandas, and Matplotlib that are essential for data manipulation and visualization.

This chapter explores how to create Python bindings for Candle using PyO3, allowing you to:
- Leverage Candle's performance advantages while working in a familiar Python environment
- Integrate Candle models with existing Python-based machine learning pipelines
- Use Python for rapid prototyping while keeping performance-critical code in Rust
- Access the rich ecosystem of Python data science tools alongside Candle

We'll cover:
- Introduction to PyO3 and how it bridges Rust and Python
- Setting up a project with PyO3 and Candle
- Creating Python bindings for Candle's core functionality
- Working with tensors across the Rust-Python boundary
- Building and training models using Python with Candle's backend
- Performance considerations and best practices
- Advanced integration patterns

By the end of this chapter, you'll be able to create Python packages that expose Candle's functionality, giving you the best of both worlds: Rust's performance and safety with Python's ease of use and rich ecosystem.

## Understanding PyO3

### What is PyO3?

PyO3 is a Rust crate that provides bindings between Rust and Python. It allows Rust code to interact with Python code and vice versa. With PyO3, you can:

1. Call Python functions from Rust
2. Call Rust functions from Python
3. Create Python modules entirely in Rust
4. Convert between Python and Rust data types

PyO3 makes it possible to write Python extension modules in Rust, which can significantly improve performance for computationally intensive tasks while maintaining the flexibility and ease of use of Python.

### How PyO3 Works

At its core, PyO3 provides a set of traits and macros that facilitate interaction with Python's C API. The key components include:

- `#[pyfunction]` - A macro for exposing Rust functions to Python
- `#[pyclass]` - A macro for exposing Rust structs as Python classes
- `#[pymethods]` - A macro for implementing Python methods on Rust structs
- `PyResult<T>` - A type for handling Python-compatible errors
- `Python<'py>` - A token representing the Python interpreter

These components work together to create a seamless bridge between Rust and Python, handling memory management, type conversions, and error propagation.

## Setting Up a PyO3 Project for Candle

### Project Structure

To create Python bindings for Candle, we'll set up a project with the following structure:

```
candle-python/
├── Cargo.toml
├── pyproject.toml
├── setup.py
├── src/
│   └── lib.rs
└── python/
    └── candle/
        ├── __init__.py
        └── examples/
            └── simple_nn.py
```

This structure separates the Rust code (in `src/`) from the Python package (in `python/`), making it easier to maintain and distribute.

### Cargo.toml Configuration

First, let's set up the `Cargo.toml` file with the necessary dependencies:

```toml
[package]
name = "candle-python"
version = "0.1.0"
edition = "2021"

[lib]
name = "candle_python"
crate-type = ["cdylib"]

[dependencies]
candle-core = "0.9.1"
candle-nn = "0.9.1"
numpy = "0.18"
pyo3 = { version = "0.18", features = ["extension-module", "abi3-py38"] }
```

Key points:
- We specify `crate-type = ["cdylib"]` to build a dynamic library that can be loaded by Python
- We include both `candle-core` and `candle-nn` as dependencies
- We add `numpy` for interoperability with NumPy arrays
- We include `pyo3` with features for extension modules and Python 3.8+ compatibility

### Python Package Configuration

Next, we'll set up the Python package configuration in `pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "candle-python"
version = "0.1.0"
description = "Python bindings for the Candle deep learning framework"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.dependencies]
numpy = ">=1.20.0"

[tool.maturin]
features = ["pyo3/extension-module"]
```

We're using Maturin, a build system for PyO3 projects, to handle the compilation and packaging of our Rust code as a Python module.

## Creating Basic Python Bindings for Candle

### Exposing Tensor Operations

Let's start by creating basic bindings for Candle's tensor operations. Here's how we might implement this in `src/lib.rs`:

```rust
use candle_core::{Device, DType, Result, Tensor};
use numpy::PyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass(name = "Tensor")]
struct PyTensor {
    tensor: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: &PyAny, device: Option<&str>) -> PyResult<Self> {
        // Convert from NumPy array
        if let Ok(array) = data.downcast::<PyArray<f32, _>>() {
            let device = match device {
                Some("cuda") => Device::Cuda(0),
                Some("cpu") => Device::Cpu,
                Some(d) => return Err(PyValueError::new_err(format!("Unknown device: {}", d))),
                None => Device::Cpu,
            };
            
            let shape: Vec<usize> = array.shape().iter().map(|&x| x as usize).collect();
            let data_slice = unsafe { array.as_slice()? };
            
            let tensor = Tensor::from_vec(data_slice.to_vec(), shape, &device)
                .map_err(|e| PyValueError::new_err(format!("Failed to create tensor: {}", e)))?;
                
            Ok(PyTensor { tensor })
        } else {
            Err(PyValueError::new_err("Expected a NumPy array"))
        }
    }
    
    #[staticmethod]
    fn zeros(shape: Vec<usize>, device: Option<&str>) -> PyResult<Self> {
        let device = match device {
            Some("cuda") => Device::Cuda(0),
            Some("cpu") => Device::Cpu,
            Some(d) => return Err(PyValueError::new_err(format!("Unknown device: {}", d))),
            None => Device::Cpu,
        };
        
        let tensor = Tensor::zeros(shape, DType::F32, &device)
            .map_err(|e| PyValueError::new_err(format!("Failed to create zeros tensor: {}", e)))?;
            
        Ok(PyTensor { tensor })
    }
    
    #[staticmethod]
    fn ones(shape: Vec<usize>, device: Option<&str>) -> PyResult<Self> {
        let device = match device {
            Some("cuda") => Device::Cuda(0),
            Some("cpu") => Device::Cpu,
            Some(d) => return Err(PyValueError::new_err(format!("Unknown device: {}", d))),
            None => Device::Cpu,
        };
        
        let tensor = Tensor::ones(shape, DType::F32, &device)
            .map_err(|e| PyValueError::new_err(format!("Failed to create ones tensor: {}", e)))?;
            
        Ok(PyTensor { tensor })
    }
    
    #[staticmethod]
    fn randn(shape: Vec<usize>, mean: f64, std: f64, device: Option<&str>) -> PyResult<Self> {
        let device = match device {
            Some("cuda") => Device::Cuda(0),
            Some("cpu") => Device::Cpu,
            Some(d) => return Err(PyValueError::new_err(format!("Unknown device: {}", d))),
            None => Device::Cpu,
        };
        
        let tensor = Tensor::randn(mean, std, shape, &device)
            .map_err(|e| PyValueError::new_err(format!("Failed to create random tensor: {}", e)))?;
            
        Ok(PyTensor { tensor })
    }
    
    fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }
    
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        // Move tensor to CPU if it's not already there
        let cpu_tensor = if self.tensor.device().is_cpu() {
            self.tensor.clone()
        } else {
            self.tensor.to_device(&Device::Cpu)
                .map_err(|e| PyValueError::new_err(format!("Failed to move tensor to CPU: {}", e)))?
        };
        
        let shape = cpu_tensor.shape();
        let data = cpu_tensor.to_vec1::<f32>()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert tensor to vec: {}", e)))?;
            
        // Create NumPy array from data
        let np = py.import("numpy")?;
        let array = np.call_method1("array", (data,))?;
        let reshaped = array.call_method1("reshape", (shape,))?;
        
        Ok(reshaped)
    }
    
    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.tensor.add(&other.tensor)
            .map_err(|e| PyValueError::new_err(format!("Addition failed: {}", e)))?;
            
        Ok(PyTensor { tensor: result })
    }
    
    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.tensor.mul(&other.tensor)
            .map_err(|e| PyValueError::new_err(format!("Multiplication failed: {}", e)))?;
            
        Ok(PyTensor { tensor: result })
    }
    
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        let result = self.tensor.matmul(&other.tensor)
            .map_err(|e| PyValueError::new_err(format!("Matrix multiplication failed: {}", e)))?;
            
        Ok(PyTensor { tensor: result })
    }
    
    fn relu(&self) -> PyResult<PyTensor> {
        let result = self.tensor.relu()
            .map_err(|e| PyValueError::new_err(format!("ReLU failed: {}", e)))?;
            
        Ok(PyTensor { tensor: result })
    }
    
    fn sum(&self, dim: Option<usize>, keep_dim: Option<bool>) -> PyResult<PyTensor> {
        let result = match dim {
            Some(d) => self.tensor.sum(d, keep_dim.unwrap_or(false))
                .map_err(|e| PyValueError::new_err(format!("Sum failed: {}", e)))?,
            None => self.tensor.sum_all()
                .map_err(|e| PyValueError::new_err(format!("Sum failed: {}", e)))?,
        };
        
        Ok(PyTensor { tensor: result })
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Tensor(shape={:?}, device={})", 
            self.tensor.shape(),
            if self.tensor.device().is_cpu() { "cpu" } else { "cuda" }
        ))
    }
}

#[pymodule]
fn candle_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    Ok(())
}
```

This code:
1. Creates a `PyTensor` class that wraps Candle's `Tensor`
2. Provides methods for creating tensors (zeros, ones, randn)
3. Implements basic operations (add, mul, matmul)
4. Adds conversion to/from NumPy arrays
5. Exposes the module as `candle_python`

### Creating a Python Module

Now, let's create a Python module that imports our Rust extension. In `python/candle/__init__.py`:

```python
from candle_python import Tensor

__all__ = ["Tensor"]
```

This simple file re-exports the `Tensor` class from our Rust extension, making it available to Python users.

## Building Neural Network Models

### Exposing Candle-NN Functionality

Next, let's expose some of Candle's neural network functionality. We'll add to our `src/lib.rs` file:

```rust
use candle_nn::{Linear, Module, VarBuilder};
use pyo3::types::PyDict;

#[pyclass(name = "Linear")]
struct PyLinear {
    linear: Linear,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let linear = Linear::new(
            vb.pp("linear").get((out_features, in_features), "weight")
                .map_err(|e| PyValueError::new_err(format!("Failed to create weight: {}", e)))?,
            if bias.unwrap_or(true) {
                Some(vb.pp("linear").get(out_features, "bias")
                    .map_err(|e| PyValueError::new_err(format!("Failed to create bias: {}", e)))?)
            } else {
                None
            },
        );
        
        Ok(PyLinear { linear })
    }
    
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.linear.forward(&input.tensor)
            .map_err(|e| PyValueError::new_err(format!("Forward pass failed: {}", e)))?;
            
        Ok(PyTensor { tensor: output })
    }
}

#[pyclass(name = "SimpleNN")]
struct PySimpleNN {
    fc1: Linear,
    fc2: Linear,
}

#[pymethods]
impl PySimpleNN {
    #[new]
    fn new(in_features: usize, hidden_size: usize, out_features: usize) -> PyResult<Self> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let fc1 = Linear::new(
            vb.pp("fc1").get((hidden_size, in_features), "weight")
                .map_err(|e| PyValueError::new_err(format!("Failed to create fc1 weight: {}", e)))?,
            Some(vb.pp("fc1").get(hidden_size, "bias")
                .map_err(|e| PyValueError::new_err(format!("Failed to create fc1 bias: {}", e)))?),
        );
        
        let fc2 = Linear::new(
            vb.pp("fc2").get((out_features, hidden_size), "weight")
                .map_err(|e| PyValueError::new_err(format!("Failed to create fc2 weight: {}", e)))?,
            Some(vb.pp("fc2").get(out_features, "bias")
                .map_err(|e| PyValueError::new_err(format!("Failed to create fc2 bias: {}", e)))?),
        );
        
        Ok(PySimpleNN { fc1, fc2 })
    }
    
    fn forward(&self, x: &PyTensor) -> PyResult<PyTensor> {
        let x = self.fc1.forward(&x.tensor)
            .map_err(|e| PyValueError::new_err(format!("FC1 forward failed: {}", e)))?;
            
        let x = x.relu()
            .map_err(|e| PyValueError::new_err(format!("ReLU failed: {}", e)))?;
            
        let x = self.fc2.forward(&x)
            .map_err(|e| PyValueError::new_err(format!("FC2 forward failed: {}", e)))?;
            
        Ok(PyTensor { tensor: x })
    }
}

#[pymodule]
fn candle_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PySimpleNN>()?;
    Ok(())
}
```

Now we've added:
1. A `Linear` layer class
2. A simple neural network model with two linear layers and a ReLU activation

Let's update our Python module in `python/candle/__init__.py`:

```python
from candle_python import Tensor, Linear, SimpleNN

__all__ = ["Tensor", "Linear", "SimpleNN"]
```

### Example: Building a Simple Neural Network in Python

Let's create an example that demonstrates how to use our Python bindings. In `python/candle/examples/simple_nn.py`:

```python
import numpy as np
from candle import Tensor, SimpleNN

# Create a simple neural network
model = SimpleNN(in_features=2, hidden_size=10, out_features=1)

# Create input data
x = Tensor(np.array([[0.5, 0.1], [0.2, 0.8], [0.9, 0.3]], dtype=np.float32))

# Forward pass
y = model.forward(x)

# Convert result back to NumPy
result = y.to_numpy()
print("Input shape:", x.shape())
print("Output shape:", y.shape())
print("Result:", result)
```

This example:
1. Creates a simple neural network with 2 input features, 10 hidden neurons, and 1 output
2. Creates input data using a NumPy array
3. Performs a forward pass through the network
4. Converts the result back to a NumPy array for display

## Advanced Integration: Training Models

### Implementing Optimizers and Loss Functions

To enable training, we need to expose optimizers and loss functions. Let's add them to our Rust code:

```rust
use candle_nn::{loss, Optimizer, SGD};

#[pyfunction]
fn mse_loss(prediction: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let loss = loss::mse(&prediction.tensor, &target.tensor)
        .map_err(|e| PyValueError::new_err(format!("MSE loss calculation failed: {}", e)))?;
        
    Ok(PyTensor { tensor: loss })
}

#[pyclass(name = "SGD")]
struct PySGD {
    optimizer: SGD,
}

#[pymethods]
impl PySGD {
    #[new]
    fn new(learning_rate: f64) -> Self {
        PySGD {
            optimizer: SGD::new(learning_rate),
        }
    }
    
    fn step(&mut self, tensors: Vec<&PyTensor>) -> PyResult<()> {
        let mut params = Vec::new();
        for tensor in tensors {
            params.push(&tensor.tensor);
        }
        
        self.optimizer.step(&params)
            .map_err(|e| PyValueError::new_err(format!("Optimizer step failed: {}", e)))?;
            
        Ok(())
    }
    
    fn zero_grad(&mut self) -> PyResult<()> {
        self.optimizer.zero_grad()
            .map_err(|e| PyValueError::new_err(format!("Zero grad failed: {}", e)))?;
            
        Ok(())
    }
}

#[pymodule]
fn candle_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyLinear>()?;
    m.add_class::<PySimpleNN>()?;
    m.add_class::<PySGD>()?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    Ok(())
}
```

And update our Python module:

```python
from candle_python import Tensor, Linear, SimpleNN, SGD, mse_loss

__all__ = ["Tensor", "Linear", "SimpleNN", "SGD", "mse_loss"]
```

### Example: Training a Model in Python

Now let's create an example that demonstrates training a model:

```python
import numpy as np
from candle import Tensor, SimpleNN, SGD, mse_loss

# Create training data for a simple regression problem: y = 2*x1 + 3*x2
np.random.seed(42)
X = np.random.rand(100, 2).astype(np.float32)
y_true = (2 * X[:, 0] + 3 * X[:, 1]).reshape(-1, 1).astype(np.float32)

# Convert to Candle tensors
X_tensor = Tensor(X)
y_tensor = Tensor(y_true)

# Create model and optimizer
model = SimpleNN(in_features=2, hidden_size=10, out_features=1)
optimizer = SGD(learning_rate=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X_tensor)
    
    # Compute loss
    loss = mse_loss(y_pred, y_tensor)
    loss_value = loss.to_numpy().item()
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step([param for param in model.parameters()])
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value:.4f}")

# Test the model
test_X = np.array([[0.5, 0.5]], dtype=np.float32)
test_X_tensor = Tensor(test_X)
prediction = model.forward(test_X_tensor)
print(f"Prediction for [0.5, 0.5]: {prediction.to_numpy().item():.4f}")
print(f"Expected: {2*0.5 + 3*0.5:.4f}")
```

This example:
1. Creates synthetic training data for a simple regression problem
2. Converts the data to Candle tensors
3. Creates a model and optimizer
4. Implements a training loop with forward pass, loss calculation, and optimization
5. Tests the trained model on new data

## Performance Considerations

When using Candle from Python, there are several performance considerations to keep in mind:

### Data Transfer Overhead

Converting between NumPy arrays and Candle tensors involves copying data, which can be expensive for large tensors. To minimize this overhead:

1. Batch your operations to reduce the number of conversions
2. Keep data in Candle tensors as much as possible during computation
3. Only convert back to NumPy when necessary (e.g., for visualization or saving results)

### GPU Utilization

Candle can leverage GPU acceleration, which can significantly improve performance. When using Candle from Python:

1. Explicitly specify the device when creating tensors
2. Keep tensors on the same device to avoid unnecessary transfers
3. Use batch processing to maximize GPU utilization

### Python GIL Limitations

Python's Global Interpreter Lock (GIL) can limit parallelism. To mitigate this:

1. Perform computationally intensive operations in Rust
2. Use Candle's built-in parallelism features
3. Consider using multiple processes for data loading and preprocessing

## Advanced Usage Patterns

### Working with Pretrained Models

One powerful use case for Python bindings is loading and using pretrained models:

```rust
#[pyfunction]
fn load_pretrained_model(model_path: &str) -> PyResult<PySimpleNN> {
    // Load weights from a file
    let device = Device::Cpu;
    let vb = VarBuilder::from_file(model_path, DType::F32, &device)
        .map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
    
    // Create model with loaded weights
    let fc1 = Linear::new(
        vb.pp("fc1").get((10, 2), "weight")
            .map_err(|e| PyValueError::new_err(format!("Failed to load fc1 weight: {}", e)))?,
        Some(vb.pp("fc1").get(10, "bias")
            .map_err(|e| PyValueError::new_err(format!("Failed to load fc1 bias: {}", e)))?),
    );
    
    let fc2 = Linear::new(
        vb.pp("fc2").get((1, 10), "weight")
            .map_err(|e| PyValueError::new_err(format!("Failed to load fc2 weight: {}", e)))?,
        Some(vb.pp("fc2").get(1, "bias")
            .map_err(|e| PyValueError::new_err(format!("Failed to load fc2 bias: {}", e)))?),
    );
    
    Ok(PySimpleNN { fc1, fc2 })
}
```

### Integration with Python ML Ecosystem

You can integrate Candle with popular Python libraries:

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from candle import Tensor, SimpleNN, SGD, mse_loss

# Load data with pandas
data = pd.read_csv("data.csv")
X = data[["feature1", "feature2"]].values.astype(np.float32)
y = data["target"].values.reshape(-1, 1).astype(np.float32)

# Split data with scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to Candle tensors
X_train_tensor = Tensor(X_train)
y_train_tensor = Tensor(y_train)
X_test_tensor = Tensor(X_test)
y_test_tensor = Tensor(y_test)

# Create and train model
model = SimpleNN(in_features=2, hidden_size=10, out_features=1)
optimizer = SGD(learning_rate=0.01)

# Training loop (omitted for brevity)
# ...

# Evaluate model
y_pred = model.forward(X_test_tensor)
y_pred_np = y_pred.to_numpy()
y_test_np = y_test_tensor.to_numpy()

# Visualize results with matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(y_test_np, y_pred_np)
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.show()
```

## Building and Distributing Your Package

### Building with Maturin

To build your Python package, you can use Maturin:

```bash
# Install Maturin
pip install maturin

# Build the package (development mode)
maturin develop

# Build the package for distribution
maturin build --release
```

### Installing the Package

Users can install your package using pip:

```bash
# Install from PyPI (if published)
pip install candle-python

# Install from a wheel file
pip install candle_python-0.1.0-cp38-cp38-manylinux_2_17_x86_64.whl
```

### Publishing to PyPI

To make your package available to others, you can publish it to PyPI:

```bash
# Build the package
maturin build --release

# Upload to PyPI
twine upload target/wheels/candle_python-0.1.0-*.whl
```

## Conclusion

In this chapter, we've explored how to create Python bindings for Candle using PyO3. By bridging these two worlds, we can leverage the performance and safety of Rust while enjoying the ease of use and rich ecosystem of Python.

The approach we've outlined allows you to:
- Create efficient deep learning models in Rust with Candle
- Expose these models to Python for integration with existing workflows
- Use Python's data science tools for preprocessing and visualization
- Achieve better performance than pure Python implementations

While there is some overhead in crossing the language boundary, the benefits often outweigh the costs, especially for computationally intensive tasks where Rust's performance shines.

As you develop your own Python bindings for Candle, remember to:
- Keep the API Pythonic and intuitive
- Minimize data transfers between languages
- Leverage Rust for performance-critical code
- Use Python for rapid prototyping and visualization

With these principles in mind, you can create powerful deep learning applications that combine the best of both languages.

## Exercises

1. Extend the Python bindings to support more tensor operations (e.g., convolution, pooling)
2. Create bindings for a convolutional neural network model
3. Implement a data loader that efficiently transfers data between NumPy and Candle
4. Build a complete image classification example using Candle from Python
5. Profile the performance of your Python bindings and identify bottlenecks