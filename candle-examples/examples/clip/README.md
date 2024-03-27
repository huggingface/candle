Contrastive Language-Image Pre-Training 

Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
pairs of images with related texts.

https://github.com/openai/CLIP

https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip



## Running on an example on cpu

```
$ cargo run --example clip --release -- --images "candle-examples/examples/clip/assets/cats.jpg","candle-examples/examples/clip/assets/ferris.jpeg" --cpu --sequences "a painting of birds","a photo of two cats","a mechanical ferris" 


Results for image: candle-examples/examples/clip/assets/cats.jpg

INFO clip: Probability: 0.001% Text: a painting of birds 
INFO clip: Probability: 99.997% Text: a photo of two cats 
INFO clip: Probability: 0.002% Text: a mechanical ferris 
INFO clip: 

Results for image: candle-examples/examples/clip/assets/ferris.jpeg

INFO clip: Probability: 0.047% Text: a painting of birds 
INFO clip: Probability: 0.100% Text: a photo of two cats 
INFO clip: Probability: 99.853% Text: a mechanical ferris 
```


## Running on an example with metal feature (mac)

```
$ cargo run --features metal --example clip --release -- --images "candle-examples/examples/clip/assets/cats.jpg","candle-examples/examples/clip/assets/ferris.jpeg" --cpu --sequences "a painting of birds","a photo of two cats","a mechanical ferris" 

Results for image: candle-examples/examples/clip/assets/cats.jpg

INFO clip: Probability: 0.001% Text: a painting of birds 
INFO clip: Probability: 99.997% Text: a photo of two cats 
INFO clip: Probability: 0.002% Text: a mechanical ferris 
INFO clip: 

Results for image: candle-examples/examples/clip/assets/ferris.jpeg

INFO clip: Probability: 0.047% Text: a painting of birds 
INFO clip: Probability: 0.100% Text: a photo of two cats 
INFO clip: Probability: 99.853% Text: a mechanical ferris
```
