import numpy as np

class cbf():
    def __init__(self, c_a, a, b, m, g, l):
        self.c_a = c_a
        self.a = a
        self.b = b
        self.m = m
        self.g = g
        self.l = l

    def forward(self, x):
        A = np.array([
            [1/(self.a**2), 0.5/(self.a*self.b)],
            [0.5/(self.a*self.b), 1/(self.b**2)]
        ])
        return 1 - np.einsum('bd,dh,bh->b', x, A, x)