# candle-whisper: speech recognition

An implementation of [OpenAI Whisper](https://github.com/openai/whisper) using
candle. Whisper is a general purpose speech recognition model, it can be used to
convert audio files (in the `.wav` format) to text. Supported features include
language detection as well as multilingual speech recognition.

## Running some example

If no audio file is passed as input, a [sample
file](https://huggingface.co/datasets/Narsil/candle-examples/resolve/main/samples_jfk.wav) is automatically downloaded
from the hub.

```bash
 cargo run --example whisper --release --features="symphonia"

> No audio file submitted: Downloading https://huggingface.co/datasets/Narsil/candle_demo/blob/main/samples_jfk.wav
> loaded wav data: Header { audio_format: 1, channel_count: 1, sampling_rate: 16000, bytes_per_second: 32000, bytes_per_sample: 2, bits_per_sample: 16 }
> pcm data loaded 176000
> loaded mel: [1, 80, 3000]
> 0.0s -- 30.0s:  And so my fellow Americans ask not what your country can do for you ask what you can do for your country
 ```

 In order to use the multilingual mode, specify a multilingual model via the
 `--model` flag, see the details below.

## Command line flags

- `--input`: the audio file to be converted to text, in wav format.
- `--language`: force the language to some specific value rather than being
  detected, e.g. `en`.
- `--task`: the task to be performed, can be `transcribe` (return the text data
  in the original language) or `translate` (translate the text to English). 
- `--timestamps`: enable the timestamp mode where some timestamps are reported
  for each recognized audio extracts.
- `--model`: the model to be used. Models that do not end with `-en` are
  multilingual models, other ones are English only models. The supported OpenAI 
  Whisper models are `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`,
  `medium`, `medium.en`, `large`, `large-v2` and `large-v3`. The supported 
  Distil-Whisper models are `distil-medium.en`, `distil-large-v2` and `distil-large-v3`.
