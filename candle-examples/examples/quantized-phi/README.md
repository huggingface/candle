# candle-quantized-phi

Candle implementation of various quantized Phi models.

## Running an example

```bash
$ cargo run --example quantized-phi --release -- --prompt "The best thing about coding in rust is "

> - it's memory safe (without you having to worry too much) 
> - the borrow checker is really smart and will catch your mistakes for free, making them show up as compile errors instead of segfaulting in runtime.
> 
> This alone make me prefer using rust over c++ or go, python/Cython etc.
> 
> The major downside I can see now: 
> - it's slower than other languages (viz: C++) and most importantly lack of libraries to leverage existing work done by community in that language. There are so many useful machine learning libraries available for c++, go, python etc but none for Rust as far as I am aware of on the first glance. 
> - there aren't a lot of production ready projects which also makes it very hard to start new one (given my background)
> 
> Another downside:
```