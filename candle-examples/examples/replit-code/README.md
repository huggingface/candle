# candle-replit-code: code completion specialized model.

[replit-code-v1_5-3b](https://huggingface.co/replit/replit-code-v1_5-3b) is a
language model specialized for code completion. This model uses 3.3B parameters
in `bfloat16` (so the GPU version will only work on recent nvidia cards).

## Running some example

```bash
cargo run --example replit-code --release -- --prompt 'def fibonacci(n): '
```
This produces the following output.

```
def fibonacci(n):  # write Fibonacci series up to n
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()


def fibonacci_loop(n):  # write Fibonacci series up to n
    """Print a Fibonacci series up to n."""
    result = []
    a, b = 0, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result


def fibonacci_generator(n):  # write Fibonacci series up to n
    """Print a Fibonacci series up to n."""
    a, b = 0, 1
    while a < n:
        yield a
        a, b = b, a+b
```
