Contrastive Language-Image Pre-Training 

Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
pairs of images with related texts.

https://github.com/openai/CLIP

https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip



## Running on an example

```
$ cargo run --example clip --release -- --image candle-examples/examples/clip/assets/cats.jpg --cpu --sequences "a painting of birds","a photo of two cats","a mechanical ferris" 

softmax_image_vec: [1.2592043e-5, 0.99996996, 1.7392886e-5]
0: 0.0012592043 "a painting of birds"
1: 99.996994 "a photo of two cats"
2: 0.0017392886 "a mechanical ferris"
```

```
$ cargo run --example clip --release -- --image candle-examples/examples/clip/assets/ferris.jpeg --cpu --sequences "a painting of birds","a photo of two cats","a mechanical ferris" 

softmax_image_vec: [0.00046970756, 0.0009994669, 0.99853086]
0: 0.046970755 "a painting of birds"
1: 0.099946685 "a photo of two cats"
2: 99.85309 "a mechanical ferris"
```





