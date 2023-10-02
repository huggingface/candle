# Simplified

## How its works

This program implements a neural network to predict the winner of the second round of elections based on the results of the first round.

Basic moments:

1. A multilayer perceptron with two hidden layers is used. The first hidden layer has 4 neurons, the second has 2 neurons.
2. The input is a vector of 2 numbers - the percentage of votes for the first and second candidates in the first stage.
3. The output is the number 0 or 1, where 1 means that the first candidate will win in the second stage, 0 means that he will lose.
4. For training, samples with real data on the results of the first and second stages of different elections are used.
5. The model is trained by backpropagation using gradient descent and the cross-entropy loss function.
6. Model parameters (weights of neurons) are initialized randomly, then optimized during training.
7. After training, the model is tested on a deferred sample to evaluate the accuracy.
8. If the accuracy on the test set is below 100%, the model is considered underfit and the learning process is repeated.

Thus, this neural network learns to find hidden relationships between the results of the first and second rounds of voting in order to make predictions for new data.


```rust,ignore
{{#include ../simplified.rs:book_training_simplified1}}
```

```rust,ignore
{{#include ../simplified.rs:book_training_simplified2}}
```

```rust,ignore
{{#include ../simplified.rs:book_training_simplified3}}
```


## Example output

```bash
Trying to train neural network.
Epoch:   1 Train loss:  4.42555 Test accuracy:  0.00%
Epoch:   2 Train loss:  0.84677 Test accuracy: 33.33%
Epoch:   3 Train loss:  2.54335 Test accuracy: 33.33%
Epoch:   4 Train loss:  0.37806 Test accuracy: 33.33%
Epoch:   5 Train loss:  0.36647 Test accuracy: 100.00%
real_life_votes: [13, 22]
neural_network_prediction_result: 0.0
```
