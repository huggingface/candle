# candle-mobileclip

MobileCLIP is family of efficient CLIP-like models using FastViT-based image encoders.

See [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://arxiv.org/abs/2311.17049)


## Running on an example on cpu

```
$ cargo run --example mobileclip --release -- --images "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg","candle-examples/examples/yolo-v8/assets/bike.jpg" --cpu --sequences  "a cycling race","a photo of two cats","a robot holding a candle"

softmax_image_vec: [2.4819004e-5, 3.81081e-6, 0.9999714, 0.9999738, 2.382714e-5, 2.3317718e-6]


Results for image: candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg

Probability: 0.0025% Text: a cycling race
Probability: 0.0004% Text: a photo of two cats
Probability: 99.9971% Text: a robot holding a candle


Results for image: candle-examples/examples/yolo-v8/assets/bike.jpg

Probability: 99.9974% Text: a cycling race
Probability: 0.0024% Text: a photo of two cats
Probability: 0.0002% Text: a robot holding a candle
```
