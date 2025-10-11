# candle-quantized-gemma

Candle implementation of quantized Gemma.

## Running an example

```bash
$ cargo run --example quantized-gemma -- --prompt "Write a function to calculate fibonacci numbers. "

> ```python
> def fibonacci(n):
>     """Calculates the nth Fibonacci number using recursion."""
>     if n <= 1:
>         return n
>     else:
>         return fibonacci(n-1) + fibonacci(n-2
> ```
```