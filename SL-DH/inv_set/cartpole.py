import numpy as np

class basic():
    def __init__(self, x_bar, d, v, theta, theta_dot):
        self.x_bar = x_bar
        self.d = d
        self.v = v
        self.theta = theta
        self.theta_dot = theta_dot

    def forward(self, states):
        if len(states.shape) == 1:
            states = np.array([states])
        x_bar, d, v, theta, theta_dot = self.x_bar, self.d, self.v, self.theta, self.theta_dot
        xleft = (states[:, 0] <= -x_bar+d) & (states[:, 0] >= -x_bar)
        xmid = (states[:, 0] < x_bar-d) & (states[:, 0] > -x_bar+d)
        xright = (states[:, 0] <= x_bar) & (states[:, 0] >= x_bar-d)
        x_dotleft = states[:, 1] >= 0
        x_dotmid = (states[:, 1] <= v) & (states[:, 1] >= -v)
        x_dotright = states[:, 1] <= 0
        x_x_dot = (xleft & x_dotleft) | (xmid & x_dotmid) | (xright & x_dotright)
        theta = (states[:, 2] <= theta) & (states[:, 2] >= -theta)
        theta_dot = (states[:, 3] <= theta_dot) & (states[:, 3] >= -theta_dot)
        return np.where(x_x_dot & theta & theta_dot, 1, -1)
    
class trap():
    def __init__(self, x_bar, m):
        self.x_bar = x_bar
        self.m = m

    def forward(self, states):
        if len(states.shape) == 1:
            states = np.array([states])
        x_bar = self.x_bar
        m = self.m
        x, x_dot = states[:, 0], states[:, 1]
        y1, y2 = m*(x + x_bar), m*(x - x_bar)
        return np.where((x_dot >= y1) & (x_dot <= y2) & (x <= x_bar) & (x >= -x_bar), 1, -1)