## SigLIP

SigLIP is multi-modal text-vision model that improves over CLIP by using a sigmoid based loss,
[HuggingFace](https://huggingface.co/google/siglip-base-patch16-224).

### Running an example
```
$ cargo run --features cuda -r --example siglip
softmax_image_vec: [2.1912122e-14, 2.3624872e-14, 1.0, 1.0, 2.4787932e-8, 3.2784535e-12]


Results for image: candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg

Probability: 0.0000% Text: a cycling race 
Probability: 0.0000% Text: a photo of two cats 
Probability: 100.0000% Text: a robot holding a candle 


Results for image: candle-examples/examples/yolo-v8/assets/bike.jpg

Probability: 100.0000% Text: a cycling race 
Probability: 0.0000% Text: a photo of two cats 
Probability: 0.0000% Text: a robot holding a candle 
```
