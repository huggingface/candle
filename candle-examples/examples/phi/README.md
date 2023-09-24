# candle-phi: 1.3b LLM with state of the art performance for <10b models.

[Phi-1.5](https://huggingface.co/microsoft/phi-1_5) is a language model using
only 1.3 billion parameters but with state of the art performance compared to
models with up to 10 billion parameters.

The candle implementation provides both the standard version as well as a
quantized variant.

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

$ cargo run --example phi --release -- \
  --prompt "Explain how to find the median of an array and write the corresponding python function.\nAnswer:" \
  --quantized --sample-len 200

Explain how to find the median of an array and write the corresponding python function.
Answer: The median is the middle value in an array. If the array has an even number of elements, the median is the average of the two middle values.

def median(arr):
    arr.sort()
    n = len(arr)
    if n % 2 == 0:
        return (arr[n//2 - 1] + arr[n//2]) / 2
    else:
        return arr[n//2]
```
