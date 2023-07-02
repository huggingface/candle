import candle

t = candle.Tensor(42.0)
print(t)
print("shape", t.shape, t.rank)
print(t + t)

t = candle.Tensor([3.0, 1, 4, 1, 5, 9, 2, 6])
print(t)
print(t+t)
t = t.reshape([2, 4])
print(t.matmul(t.t()))
