# candle-phi: 1.3b and 2.7b LLM with state of the art performance for <10b models.

[Phi-1.5](https://huggingface.co/microsoft/phi-1_5), 
[Phi-2](https://huggingface.co/microsoft/phi-2), and
[Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) are language models using
only 1.3, 2.7, and 3.8 billion parameters but with state of the art performance compared to
models with up to 10 billion parameters.

The candle implementation provides both the standard version as well as a
quantized variant.

## Running some examples

For the v2 version.
```bash
$ cargo run --example phi --release -- --model 2 \
  --prompt "A skier slides down a frictionless slope of height 40m and length 80m. What's the skier speed at the bottom?"

A skier slides down a frictionless slope of height 40m and length 80m. What's the skier speed at the bottom?

Solution:
The potential energy of the skier is converted into kinetic energy as it slides down the slope. The formula for potential energy is mgh, where m is mass, g is acceleration due to gravity (9.8 m/s^2), and h is height. Since there's no friction, all the potential energy is converted into kinetic energy at the bottom of the slope. The formula for kinetic energy is 1/2mv^2, where v is velocity. We can equate these two formulas:
mgh = 1/2mv^2
Solving for v, we get:
v = sqrt(2gh)
Substituting the given values, we get:
v = sqrt(2*9.8*40) = 28 m/s
Therefore, the skier speed at the bottom of the slope is 28 m/s.
```

For the v1.5 version.
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

This also supports the [Puffin Phi v2
model](https://huggingface.co/teknium/Puffin-Phi-v2) for human interaction.
```
$ cargo run --example phi --release  -- \
    --prompt "USER: What would you do on a sunny day in Paris?\nASSISTANT:" \
    --sample-len 200 --model puffin-phi-v2 --quantized 
USER: What would you do on a sunny day in Paris?
ASSISTANT: On a sunny day in Paris, you could visit the Musée du Louvre to admire the famous
painting "Mona Lisa" by Leonardo da Vinci. You might also want to stroll along the Champs-Élysées
and enjoy the beautiful architecture of the buildings around you. Don't forget to stop by a café
for a cup of coffee and to soak up the sun!"
```
