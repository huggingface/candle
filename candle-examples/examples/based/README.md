# candle-based

Experimental, not instruction-tuned small LLM from the Hazy Research group, combining local and linear attention layers.

[Blogpost](https://hazyresearch.stanford.edu/blog/2024-03-03-based)

[Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)

## Running an example

```bash
$ cargo run --example based --release -- --prompt "Flying monkeys are" --which 1b-50b --sample-len 100

Flying monkeys are a common sight in the wild, but they are also a threat to humans.

The new study, published today (July 31) in the journal Science Advances, shows that the monkeys are using their brains to solve the problem of how to get around the problem.

"We found that the monkeys were using a strategy called 'cognitive mapping' - they would use their brains to map out the route ahead," says lead author Dr. David J. Smith from the University of California

```
