import numpy as np
from ctypes import *

t = cdll.LoadLibrary('./test.so')
x = np.arange(1, 10, 1)

int_ = np.array(x, dtype=np.int32)
long_ = np.array(x, dtype=np.int64)
float_ = np.array(x, dtype=np.float32)

t.f_int(int_.ctypes.data_as(POINTER(c_int)), len(int_))
t.f_long(long_.ctypes.data_as(POINTER(c_long)), len(long_))
t.f_float(float_.ctypes.data_as(POINTER(c_float)), len(float_))

print(int_)
print(long_)
print(float_)

# result:
# ubuntu 16.04  x64  gcc 5.4.0
# np.int32 -> c.int
# np.int64 -> c.long
# np.float32 -> c.float
