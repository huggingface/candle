# candle-convmixer

A lightweight CNN architecture that processes image patches similar to a vision transformer, with separate spatial and channel convolutions.

ConvMixer from [Patches Are All You Need?](https://arxiv.org/pdf/2201.09792) and [ConvMixer](https://github.com/locuslab/convmixer). 

## Running an example

```bash
$ cargo run --example convmixer --release -- --image candle-examples/examples/yolo-v8/assets/bike.jpg

> mountain bike, all-terrain bike, off-roader: 61.75%
> unicycle, monocycle     : 5.73%
> moped                   : 3.66%
> bicycle-built-for-two, tandem bicycle, tandem: 3.51%
> crash helmet            : 0.85%
```
