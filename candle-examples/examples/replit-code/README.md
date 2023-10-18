# candle-replit-code: code completion specialized model.

[replit-code-v1_5-3b](https://huggingface.co/replit/replit-code-v1_5-3b) is a
language model specialized for code completion. This model uses 3.3B parameters
in `bfloat16` (so the GPU version will only work on recent nvidia cards).

## Running some example

```bash
cargo run --example replit-code --release -- --prompt 'def fibonacci(n): '
```
This produces the following output which actually doesn't generate the fibonacci
series properly.

```
def fibonacci(n):  # write Fibonacci series up to n
    """Print a Fibonacci series up to n."""

    assert type(n) == int, "n must be an integer"
    
    if (type(fib_list)==None or len==0 ):
        fib_list = [1]
        
    for i in range((len-2)):  # start at 2nd element of list and go until end. 
        n += 1
        
        print("Fibonacci number",n,"is:",i)
        
def main():
    """Call the functions."""

    userInput=input('Enter a positive integer: ')
    
    fibonacci(userInput)
    


    
    
    

if __name__ == '__main__':  # only run if this file is called directly. 
    print("This program prints out Fibonacci numbers.")
    main()
```
