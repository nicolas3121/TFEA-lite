import numpy as np

deg = 11

x, w = np.polynomial.legendre.leggauss(deg)
rule = []
# x = (1 + x) / 2
# w /= 2

for u0, w1 in zip(x, w):
    for v0, w2 in zip(x, w):
        rule.append([u0, v0, w1 * w2])

print(np.array_repr(np.array(rule, dtype=np.float64), precision=17))
