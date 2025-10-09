# candle_bert_single_file_binary_builder

This crate provides and isolates the necessary build steps to fetch the model files for the [`bert_single_file_binary` example](../examples/bert_single_file_binary/). See [https://github.com/huggingface/candle/pull/3104#issuecomment-3369276760](https://github.com/huggingface/candle/pull/3104#issuecomment-3369276760) for background.

### Limitations

1. Because the model files must be available at compile time, a special build step is needed
2. The model id and revision is hardcoded
3. The model files are downloaded from directly Hugging Face at compile time for simplicity sake, not using the hf-hub library
   1. Since the file paths must be known at compile time it is easier to download the files into the example dir than navigate the hub cache dir snapshots.
