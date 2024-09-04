import numpy as np

rnd = np.random.default_rng()
a = rnd.integers(0, 112, 10, endpoint=True)
print(a)
print(np.max(a))
a[np.where(a == a.max())] = 0
print(a)