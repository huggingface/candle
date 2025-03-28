# candle-moondream

[Moondream](https://github.com/vikhyat/moondream) is a computer-vision model can answer real-world questions about images. It's tiny by today's models, with only 1.6B parameters. That enables it to run on a variety of devices, including mobile phones and edge devices.

## Running some examples
First download an example image
```bash
$ wget https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg
```

<img src="https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg" width="200">

Now you can run Moondream from the `candle-examples` crate:
```bash
$ cargo run --example moondream --release -- --prompt "Describe the people behind the bikers?" --image "candle-examples/examples/yolo-v8/assets/bike.jpg"

avavx: false, neon: true, simd128: false, f16c: false
temp: 0.00 repeat-penalty: 1.00 repeat-last-n: 64
retrieved the files in 3.395583ms
Running on CPU, to run on GPU(metal), build this example with `--features metal`
loaded the model in 5.485493792s
loaded and encoded the image Tensor[dims 3, 378, 378; f32] in 4.801396417s
starting the inference loop
 The girl is eating a hamburger.<
9 tokens generated (0.68 token/s)
```