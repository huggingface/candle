# candle-granite LLMs from IBM Research

[Granite](https://www.ibm.com/granite) is a family of Large Language Models built for business, to help drive trust and scalability in AI-driven applications.

## Running the example

```bash
$ cargo run --example granite --features metal -r -- --model-type "granite7b-instruct" \
    --prompt "Explain how quantum computing differs from classical computing, focusing on key concepts like qubits, superposition, and entanglement. Describe two potential breakthroughs in the fields of drug discovery and cryptography. Offer a convincing argument for why businesses and governments should invest in quantum computing research now, emphasizing its future benefits and the risks of falling behind"

    Explain how quantum computing differs from classical computing, focusing on key concepts like qubits, superposition, and entanglement. Describe two potential breakthroughs in the fields of drug discovery and cryptography. Offer a convincing argument for why businesses and governments should invest in quantum computing research now, emphasizing its future benefits and the risks of falling behind competitors.

    In recent years, there has been significant interest in quantum computing due to its potential to revolutionize various fields, including drug discovery, cryptography, and optimization problems. Quantum computers, which leverage the principles of quantum mechanics, differ fundamentally from classical computers. Here are some of the key differences:
```

## Supported Models
There are two different modalities for the Granite family models: Language and Code.

### Granite for language
1. [Granite 7b Instruct](https://huggingface.co/ibm-granite/granite-7b-instruct)
