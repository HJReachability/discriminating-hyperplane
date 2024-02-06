import numpy as np

class cbf():
    def __init__(self, a, r, gamma):
        self.a = a
        self.r = r
        self.gamma = gamma

    def h(self, x):
        px, py, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return (px + (v**2)/(4*self.a)*np.cos(theta))**2 + (py + (v**2)/(4*self.a)*np.sin(theta))**2

    def L_fb(self, x):
        px, py, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        h_x = self.h(x)
        return (v/np.sqrt(h_x))*(px*np.cos(theta)+py*np.sin(theta)+(v**2)/(4*self.a))

    def L_gb(self, x):
        px, py, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        h_x = self.h(x)
        return np.array([
            ((v**2)/(4*self.a))*(py*np.cos(theta)-px*np.sin(theta))/np.sqrt(h_x),
            (v/(2*self.a))*(px*np.cos(theta)+py*np.sin(theta)+(v**2)/(4*self.a))/np.sqrt(h_x) - v/(2*self.a)
        ])

    def forward(self, x):
        if len(x.shape) == 1:
            x = np.array([x])
        px, py, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        h_x = self.h(x)
        return np.sqrt(h_x) - (self.r + v**2/(4*self.a))

    def get_hyp(self, x):
        if len(x.shape) == 1:
            x = np.array([x])
        H_x = self.H(x)
        return self.L_gb(x).reshape(-1), (-self.gamma*H_x-self.L_fb(x))