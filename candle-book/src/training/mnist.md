# MNIST

So we now have downloaded the MNIST parquet files, let's put them in a simple struct.

```rust,ignore
{{#include ../lib.rs:book_training_3}}
```

The parsing of the file and putting it into single tensors requires the dataset to fit the entire memory.
It is quite rudimentary, but simple enough for a small dataset like MNIST.
