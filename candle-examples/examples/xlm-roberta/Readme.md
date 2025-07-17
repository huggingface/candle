# candle-xlm-roberta

This example demonstrates how to use the XLM-RoBERTa model in Candle especially known for their use in reranking. It uses the `fill-mask` task to generate a word for a masked token. And a `reranker` task to rerank a list of documents for a given query.

## Usage

Fill Mask:
```bash
cargo run --example xlm-roberta --release -- --task fill-mask --model xlm-roberta-base
```
```markdown
Sentence: 0 : Hello I'm a fashion model.
Sentence: 1 : I'm a little boy.
Sentence: 2 : I'm living in berlin.
```

Reranker:
```bash
cargo run --example xlm-roberta --release -- --task reranker --model bge-reranker-base
```
```markdown
Ranking Results:
--------------------------------------------------------------------------------
> Rank #4  | Score: 0.0001 | South Korea is a country in East Asia.
> Rank #5  | Score: 0.0000 | There are forests in the mountains.
> Rank #2  | Score: 0.7314 | Pandas look like bears.
> Rank #3  | Score: 0.6948 | There are some animals with black and white fur.
> Rank #1  | Score: 0.9990 | The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.
--------------------------------------------------------------------------------
```

Text-Classification:
```bash
cargo run --example xlm-roberta -- --task text-classification --model xlmr-formality-classifier
```
```markdown
Formality Scores:
Text 1: "I like you. I love you"
  formal: 0.9933
  informal: 0.0067

Text 2: "Hey, what's up?"
  formal: 0.8812
  informal: 0.1188

Text 3: "Siema, co porabiasz?"
  formal: 0.9358
  informal: 0.0642

Text 4: "I feel deep regret and sadness about the situation in international politics."
  formal: 0.9987
  informal: 0.0013
```