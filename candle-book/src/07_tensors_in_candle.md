# 6. Tensors

## Introduction to Tensors

Tensors are multi-dimensional arrays that can represent scalars, vectors, matrices, and higher-dimensional data. They are the building blocks for all neural network operations, from simple arithmetic to complex transformations.

A tensor is characterized by:
1. **Data Type**: The type of elements it contains (e.g., f32, f64, i64)
2. **Shape**: The dimensions of the tensor (e.g., a scalar is 0D, a vector is 1D, a matrix is 2D)
3. **Device**: Where the tensor is stored (CPU or GPU)


## Creating Tensors

Candle provides several ways to create tensors. Let's explore the most common methods:

### From Scalar Values

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    // Create a tensor from a scalar value
    let scalar = Tensor::new(42f32, &Device::Cpu)?;
    println!("Scalar tensor: {}", scalar);

    Ok(())
}
```

### From Arrays

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    // Create a 1D tensor (vector) from an array
    let vector = Tensor::new(&[1f32, 2., 3., 4., 5.], &Device::Cpu)?;
    println!("Vector tensor: {}", vector);

    // Create a 2D tensor (matrix) from a 2D array
    let matrix = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    println!("Matrix tensor: {}", matrix);

    Ok(())
}
```

### Using Builder Functions

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a tensor filled with zeros
    let zeros = Tensor::zeros((2, 3), candle_core::DType::F32, &device)?;
    println!("Zeros tensor: {}", zeros);

    // Create a tensor filled with ones
    let ones = Tensor::ones((2, 3), candle_core::DType::F32, &device)?;
    println!("Ones tensor: {}", ones);

    // Create a tensor with random values
    let random = Tensor::rand(0f32, 1f32, (2, 3), &device)?;
    println!("Random tensor: {}", random);

    // Create an identity matrix
    let identity = Tensor::eye(3, candle_core::DType::F32, &device)?;
    println!("Identity tensor: {}", identity);

    // Create a tensor with a range of values
    let range = Tensor::arange(0f32, 10f32, 1f32, &device)?;
    println!("Range tensor: {}", range);

    Ok(())
}
```

### From Existing Data

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    // Create a tensor from a Vec
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, (2, 3), &Device::Cpu)?;
    println!("Tensor from Vec: {}", tensor);

    // Create a tensor from a slice
    let slice = &[7.0, 8.0, 9.0, 10.0];
    let tensor = Tensor::from_slice(slice, (2, 2), &Device::Cpu)?;
    println!("Tensor from slice: {}", tensor);

    Ok(())
}
```

## Printing Tensors

As you've seen in the examples above, tensors can be printed using the `println!` macro with the `{}` format specifier. This works because Tensor implements the `Display` trait.

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    // Create a tensor
    let tensor = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;

    // Print the tensor
    println!("Tensor: {}", tensor);

    // Print tensor with debug information
    println!("Tensor debug: {:?}", tensor);

    // Print tensor shape
    println!("Tensor shape: {:?}", tensor.shape());

    // Print tensor dtype
    println!("Tensor dtype: {:?}", tensor.dtype());

    // Print tensor device
    println!("Tensor device: {:?}", tensor.device());

    Ok(())
}
```

For large tensors, you might want to print only a subset of the values:

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    // Create a large tensor
    let large_tensor = Tensor::rand(0f32, 1f32, (10, 10), &Device::Cpu)?;

    // Print the first row
    let first_row = large_tensor.get(0)?;
    println!("First row: {}", first_row);

    // Print a specific element
    let element = large_tensor.get((0, 0))?.to_scalar::<f32>()?;
    println!("Element at (0,0): {}", element);

    Ok(())
}
```

## Shape and Reshape Operations

Tensors can be reshaped to change their dimensions while preserving the total number of elements.

### Getting the Shape

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let tensor = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;

    // Get the shape as a Vec
    let shape = tensor.shape().to_vec();
    println!("Tensor shape: {:?}", shape);

    // Get individual dimensions
    let dim0 = tensor.dim(0)?;
    let dim1 = tensor.dim(1)?;
    println!("Dimension 0: {}, Dimension 1: {}", dim0, dim1);

    // Get the total number of elements
    let numel = tensor.elem_count();
    println!("Number of elements: {}", numel);

    Ok(())
}
```

### Reshaping Tensors

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    // Create a 2x3 tensor
    let tensor = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    println!("Original tensor: {}", tensor);
    println!("Original shape: {:?}", tensor.shape());

    // Reshape to 3x2
    let reshaped = tensor.reshape((3, 2))?;
    println!("Reshaped tensor: {}", reshaped);
    println!("New shape: {:?}", reshaped.shape());

    // Reshape to 1D (flatten)
    let flattened = tensor.flatten_all()?;
    println!("Flattened tensor: {}", flattened);
    println!("Flattened shape: {:?}", flattened.shape());

    // Reshape with -1 (automatic dimension)
    let auto_reshaped = tensor.reshape((6, 1))?;
    println!("Auto-reshaped tensor: {}", auto_reshaped);
    println!("Auto-reshaped shape: {:?}", auto_reshaped.shape());

    Ok(())
}
```

### Adding and Removing Dimensions

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    // Create a 2x3 tensor
    let tensor = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    println!("Original tensor: {}", tensor);
    println!("Original shape: {:?}", tensor.shape());

    // Add a dimension (unsqueeze)
    let unsqueezed = tensor.unsqueeze(0)?;
    println!("Unsqueezed tensor: {}", unsqueezed);
    println!("Unsqueezed shape: {:?}", unsqueezed.shape());

    // Remove a dimension (squeeze)
    let squeezed = unsqueezed.squeeze(0)?;
    println!("Squeezed tensor: {}", squeezed);
    println!("Squeezed shape: {:?}", squeezed.shape());

    // Transpose (swap dimensions)
    let transposed = tensor.transpose(0, 1)?;
    println!("Transposed tensor: {}", transposed);
    println!("Transposed shape: {:?}", transposed.shape());

    Ok(())
}
```

## Linear Algebra Operations

Tensors support a wide range of linear algebra operations, the base of neural network computations.

### Basic Arithmetic

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    let b = Tensor::new(&[[7f32, 8., 9.], [10., 11., 12.]], &Device::Cpu)?;

    // Addition
    let sum = a.add(&b)?;
    println!("a: {}", a);
    println!("b: {}", b);
    println!("a + b: {}", sum);

    // Subtraction
    let diff = a.sub(&b)?;
    println!("a - b: {}", diff);

    // Multiplication (element-wise)
    let prod = a.mul(&b)?;
    println!("a * b (element-wise): {}", prod);

    // Division (element-wise)
    let quot = a.div(&b)?;
    println!("a / b (element-wise): {}", quot);

    // Scalar operations
    let scalar = 2.0;
    let scaled = a.mul_scalar(scalar)?;
    println!("a * {}: {}", scalar, scaled);

    Ok(())
}
```

### Matrix Multiplication

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;  // 2x3
    let b = Tensor::new(&[[7f32, 8.], [9., 10.], [11., 12.]], &Device::Cpu)?;  // 3x2

    // Matrix multiplication
    let matmul = a.matmul(&b)?;
    println!("a: {}", a);
    println!("b: {}", b);
    println!("a @ b (matrix multiplication): {}", matmul);
    println!("Result shape: {:?}", matmul.shape());

    // Dot product (for vectors)
    let v1 = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
    let v2 = Tensor::new(&[4f32, 5., 6.], &Device::Cpu)?;
    let dot = v1.dot(&v2)?;
    println!("v1: {}", v1);
    println!("v2: {}", v2);
    println!("v1 · v2 (dot product): {}", dot);

    Ok(())
}
```

### Advanced Linear Algebra

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a square matrix
    let matrix = Tensor::new(&[[4f32, 2., 1.], [2., 5., 3.], [1., 3., 6.]], &device)?;
    println!("Matrix: {}", matrix);

    // Compute the trace (sum of diagonal elements)
    let trace = matrix.trace()?;
    println!("Trace: {}", trace);

    // Compute the determinant
    // Note: Candle might not have a direct determinant function,
    // but it can be computed using decompositions

    // Compute the inverse (if available in Candle)
    // let inverse = matrix.inverse()?;
    // println!("Inverse: {}", inverse);

    // Compute eigenvalues and eigenvectors (if available)
    // let (eigenvalues, eigenvectors) = matrix.eig()?;
    // println!("Eigenvalues: {}", eigenvalues);
    // println!("Eigenvectors: {}", eigenvectors);

    // Compute the norm
    let norm = matrix.flatten_all()?.sqr()?.sum_all()?.sqrt()?;
    println!("Frobenius norm: {}", norm);

    Ok(())
}
```

## Broadcasting

Broadcasting is a powerful feature that allows operations between tensors of different shapes. It automatically expands smaller tensors to match the shape of larger ones, making operations more convenient.

### Understanding Broadcasting

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a 2x3 matrix
    let matrix = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
    println!("Matrix: {}", matrix);
    println!("Matrix shape: {:?}", matrix.shape());

    // Create a vector
    let vector = Tensor::new(&[10f32, 20., 30.], &device)?;
    println!("Vector: {}", vector);
    println!("Vector shape: {:?}", vector.shape());

    // Broadcasting addition
    // The vector is automatically broadcast to shape [2, 3]
    let result = matrix.add(&vector)?;
    println!("Matrix + Vector (broadcast): {}", result);

    // Create a scalar tensor
    let scalar = Tensor::new(5f32, &device)?;
    println!("Scalar: {}", scalar);

    // Broadcasting multiplication
    // The scalar is broadcast to match the matrix shape
    let scaled = matrix.mul(&scalar)?;
    println!("Matrix * Scalar (broadcast): {}", scaled);

    Ok(())
}
```

### Broadcasting with Different Dimensions

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a 3x1 matrix
    let a = Tensor::new(&[[1f32], [2.], [3.]], &device)?;
    println!("a (3x1): {}", a);

    // Create a 1x4 matrix
    let b = Tensor::new(&[[10f32, 20., 30., 40.]], &device)?;
    println!("b (1x4): {}", b);

    // Broadcasting multiplication
    // a is broadcast to [3, 4] and b is broadcast to [3, 4]
    let result = a.mul(&b)?;
    println!("a * b (broadcast to 3x4): {}", result);
    println!("Result shape: {:?}", result.shape());

    // Create a 2x3x1 tensor
    let c = Tensor::new(&[[[1f32], [2.], [3.]], [[4.], [5.], [6.]]], &device)?;
    println!("c (2x3x1): {}", c);

    // Create a 1x1x4 tensor
    let d = Tensor::new(&[[[10f32, 20., 30., 40.]]], &device)?;
    println!("d (1x1x4): {}", d);

    // Broadcasting addition
    // c is broadcast to [2, 3, 4] and d is broadcast to [2, 3, 4]
    let result = c.add(&d)?;
    println!("c + d (broadcast to 2x3x4): {}", result);
    println!("Result shape: {:?}", result.shape());

    Ok(())
}
```

## Indexing and Slicing

Tensors can be indexed and sliced to access specific elements or subsets of data.

### Basic Indexing

```rust
use candle_core::{Tensor, Device};
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a 2x3 matrix
    let tensor = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
    println!("Tensor: {}", tensor);

    // Get a single element
    let element = tensor.get((0, 1))?.to_scalar::<f32>()?;
    println!("Element at (0,1): {}", element);

    // Get a row
    let row = tensor.get(0)?;
    println!("First row: {}", row);

    // Get a column (using indexing and transpose)
    let column = tensor.transpose(0, 1)?.get(0)?;
    println!("First column: {}", column);

    Ok(())
}
```

### Slicing

```rust
use candle_core::{Tensor, Device, IndexOp};
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a 4x4 matrix
    let tensor = Tensor::new(
        &[
            [1f32, 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ],
        &device
    )?;
    println!("Tensor: {}", tensor);

    // Slice rows (get rows 1 and 2)
    let rows = tensor.i(1..3)?;
    println!("Rows 1-2: {}", rows);

    // Slice columns (get columns 1 to 3)
    let cols = tensor.i(..).i(1..4)?;
    println!("Columns 1-3: {}", cols);

    // Get a 2x2 submatrix (rows 1-2, columns 1-2)
    let submatrix = tensor.i(1..3).i(1..3)?;
    println!("Submatrix (rows 1-2, columns 1-2): {}", submatrix);

    // Get every other element
    let strided = tensor.i((.., 2))?;
    println!("Every other row: {}", strided);

    Ok(())
}
```

### Advanced Indexing

```rust
use candle_core::{Tensor, Device, IndexOp};
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Create a 3x4 matrix
    let tensor = Tensor::new(
        &[
            [1f32, 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.]
        ],
        &device
    )?;
    println!("Tensor: {}", tensor);

    // Index using another tensor
    let indices = Tensor::new(&[0, 2], &device)?;
    let selected_rows = tensor.index_select(&indices, 0)?;
    println!("Selected rows (0 and 2): {}", selected_rows);

    // Gather elements
    let row_indices = Tensor::new(&[0, 1, 2], &device)?;
    let col_indices = Tensor::new(&[1, 2, 0], &device)?;
    let gathered = tensor.gather(&row_indices, &col_indices)?;
    println!("Gathered elements [(0,1), (1,2), (2,0)]: {}", gathered);

    // Masked select
    let mask = Tensor::new(
        &[
            [true, false, true, false],
            [false, true, false, true],
            [true, false, true, false]
        ],
        &device
    )?;
    let masked = tensor.masked_select(&mask)?;
    println!("Masked select: {}", masked);

    Ok(())
}
```

## Conclusion

In this chapter, we've explored tensors in the Candle library, covering their creation, manipulation, and various operations. Tensors are the fundamental building blocks of neural networks, and understanding how to work with them is essential for implementing and understanding deep learning models.

We've seen how to:
- Create tensors from various data sources
- Print and inspect tensors
- Reshape and manipulate tensor dimensions
- Perform linear algebra operations
- Use broadcasting to simplify operations between tensors of different shapes
- Index and slice tensors to access specific data

These tensor operations form the foundation for all the neural network architectures we'll explore in subsequent chapters. As we move forward, you'll see how these basic operations combine to create powerful models capable of solving complex problems.

In the next chapter, we'll build on this foundation to explore more advanced neural network architectures and techniques.
