# candle-starcoder: code generation model

[phi-1.5](https://huggingface.co/microsoft/phi-1_5).

## Running some example

```bash
$ cargo run --example phi --release -- --prompt "def print_prime(n): "

def print_prime(n): 
    print("Printing prime numbers")
    for i in range(2, n+1):
        if is_prime(i):
            print(i)

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n))+1):
        if n % i == 0:
            return False
    return True
```
