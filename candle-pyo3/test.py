import os
import sys

# The "import candle" statement below works if there is a "candle.so" file in sys.path.
# Here we check for shared libraries that can be used in the build directory.
BUILD_DIR = "./target/release-with-debug"
so_file = BUILD_DIR + "/candle.so"
if os.path.islink(so_file): os.remove(so_file)
for lib_file in ["libcandle.dylib", "libcandle.so"]:
    lib_file_ = BUILD_DIR + "/" + lib_file
    if os.path.isfile(lib_file_):
        os.symlink(lib_file, so_file)
        sys.path.insert(0, BUILD_DIR)
        break

import candle

t = candle.Tensor(42.0)
print(t)
print(t.shape, t.rank, t.device)
print(t + t)

t = candle.Tensor([3.0, 1, 4, 1, 5, 9, 2, 6])
print(t)
print(t+t)

t = t.reshape([2, 4])
print(t.matmul(t.t()))

print(t.to_dtype(candle.u8))
print(t.to_dtype("u8"))

t = candle.randn((5, 3))
print(t)
print(t.dtype)
