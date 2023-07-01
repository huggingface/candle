import candle

t = candle.Tensor(42.0)
print(t)
print("shape", t.shape, t.rank)
print(t + t)
