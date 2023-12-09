# search-sorted

`candle` implementation of [`torch.searchsorted`](https://pytorch.org/docs/stable/generated/torch.searchsorted.html).  
**Usage Notes**:
- the tensor being search **MUST** be sorted
- works for `N-D` sorted sequence and `N-D` values where the leading dimensions must match
- if search values are `1-D` and sorted sequence is `N-D`, the search is applied to all elements of the sorted sequence and the result is `N-D` with the same shape as the search values
  - e.g., for values `[[3, 6, 9], [1, 2, 3]]` and sorted sequence is `[1, 3, 5, 7, 9]`, the result is `[[1, 2, 4], [0, 0, 1]]`
- if search values are `N-D` and sorted sequence is `1-D`, result is `N-D` with the same shape as the search values
  - e.g., for values `[3, 6, 9]` and sorted sequence is `[[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]`, the result is `[[1, 3, 4], [1, 2, 4]]` 
- supports `left` and `right` modes
  - `left` mode returns the index of the first element in the sorted sequence that is greater than or equal to the search value
  - `right` mode returns the index of the first element in the sorted sequence that is greater than the search value
  - `left` mode is the default
- does not support the `sort` parameter of `torch.searchsorted`

## Running an example

#### CPU
```
$ cargo run --example search-sorted

Running on CPU, to run on GPU, build this example with `--features cuda`
Search sorted left
t1: [[ 1.,  3.,  5.,  7.,  9.],
 [ 2.,  4.,  6.,  8., 10.]]
Tensor[[2, 5], f32]
t2: [[3., 6., 9.],
 [3., 6., 9.]]
Tensor[[2, 3], f32]
t3: [[1, 3, 4],
 [1, 2, 4]]
Tensor[[2, 3], i64]
Search sorted right
t1: [[ 1.,  3.,  5.,  7.,  9.],
 [ 2.,  4.,  6.,  8., 10.]]
Tensor[[2, 5], f32]
t2: [[3., 6., 9.],
 [3., 6., 9.]]
Tensor[[2, 3], f32]
t3: [[2, 3, 5],
 [1, 3, 4]]
Tensor[[2, 3], i64]
```
#### CUDA
```
cargo run --example search-sorted --features cuda

Search sorted left
t1: [[ 1.,  3.,  5.,  7.,  9.],
 [ 2.,  4.,  6.,  8., 10.]]
Tensor[[2, 5], f32, cuda:0]
t2: [[3., 6., 9.],
 [3., 6., 9.]]
Tensor[[2, 3], f32, cuda:0]
t3: [[1, 3, 4],
 [1, 2, 4]]
Tensor[[2, 3], i64, cuda:0]
Search sorted right
t1: [[ 1.,  3.,  5.,  7.,  9.],
 [ 2.,  4.,  6.,  8., 10.]]
Tensor[[2, 5], f32, cuda:0]
t2: [[3., 6., 9.],
 [3., 6., 9.]]
Tensor[[2, 3], f32, cuda:0]
t3: [[2, 3, 5],
 [1, 3, 4]]
Tensor[[2, 3], i64, cuda:0]
```
