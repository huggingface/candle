# candle-qwen: large language model series from Alibaba Cloud

Qwen 3 is a series of large language models that provide strong performances on English and Chinese.

- [Qwen3 Collection](https://huggingface.co/collections/Qwen/qwen3).
- [Model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).

## Running the example

```bash
$ cargo run --example qwen --release  -- --prompt "Hello there "
```

Various model sizes are available via the `--model` argument, including the MoE
variant.

```bash
$ cargo run --example qwen --release  -- --model moe-a2.7b --prompt 'def print_prime(n: int): '
def print_prime(n: int):  # n is the number of primes to be printed
    for i in range(2, n + 1):
        if all(i % j != 0 for j in range(2, i)):
            print(i)
```

The qwen3 MoE variant is also an option.

```bash
$ cargo run --example qwen --features metal --release  -- --prompt "Write a poem about butterflies. <think></think>." --model "3-moe-a3b"
> In morning's hush, where daisies sleep,  
> A fleeting dance through sunlit deep—  
> They flutter soft on gossamer thread,  
> The messengers of spring’s own head.
> 
> With painted sails and delicate grace,  
> They drift from bloom to blossom's face.  
> Each wing a tale in hues unseen,  
> Of ancient dreams and secrets between.
> 
> No sound they make, yet still they speak—  
> Of time that flies, of life so brief.  
> A fleeting kiss on summer’s breath,  
> A whisper lost before death.
> 
> Yet in their flight, the soul takes wing,  
> And for a moment, all is spring.  
> For though they fade, they never die—  
> Their beauty lives where hearts can fly.
> 161 tokens generated (3.00 token/s)
```

```shell
# Local unquantized 32B MoE model (with Fused MoE kernel) (~80GB GPU memory)
cargo run --example qwen --features cuda --release  -- --prompt "Write a poem about butterflies." --model "3-moe-a3b" --weight-path /path/Qwen3-Coder-30B-A3B-Instruct/
```

```
avx: true, neon: false, simd128: false, f16c: true
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
retrieved the files in 994.303698ms
loaded the model in 10.009002649s
Write a poem about butterflies. The poem should include the word "emerald" at least three times.


**Wings of Wonder**

In meadows where the emerald grass sways,
Butterflies dance in morning's rays,
With wings like stained glass, bright and free,
They flutter through the emerald sea.

Their delicate forms catch the light,
As if they're made of pure delight,
Each emerald hue a whispered prayer
That floats above the world with care.

From chrysalis to flight so true,
These creatures teach us how to renew,
With emerald eyes that see the grace
Of time and beauty in their space.

So let us be like butterflies,
Dancing through life's gentle skies,
Carrying emerald dreams along,
Till we're free from sorrow's song.
160 tokens generated (30.82 token/s)
```