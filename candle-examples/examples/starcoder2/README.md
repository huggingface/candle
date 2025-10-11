# candle-starcoder2

Candle implementation of Star Coder 2 family of code generation model from [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/pdf/2402.19173).

## Running an example

```bash
$ cargo run --example starcoder2 -- --prompt "write a recursive fibonacci function in python "

> # that returns the nth number in the sequence.
> 
> def fib(n):
>     if n

```