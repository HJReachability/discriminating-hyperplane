import numpy as np

"""
All classes must have batched implementations of functions
"""

class inv_ped_CBF():
  def __init__(self, c_a, a, b, m, g, l):
    self.c_a = c_a
    self.a = a
    self.b = b
    self.m = m
    self.g = g
    self.l = l

  def L_fh(self, x):
    theta = x[:, 0]
    theta_dot = x[:, 1]
    a, b, g, l = self.a, self.b, self.g, self.l
    return ((-2*theta*theta_dot)/(a**2))-((theta_dot**2)/(a*b))-np.sin(theta)*((2*theta_dot*g/(l*(b**2))) + theta*g/(l*a*b))

  def L_gh(self, x):
    theta = x[:, 0]
    theta_dot = x[:, 1]
    c_a, a, b, m, l = self.c_a, self.a, self.b, self.m, self.l
    return (-theta/(m*(l**2)*a*b)) - (2*theta_dot/(m*(l**2)*(b**2)))

  def neg_alpha(self, x):
    theta = x[:, 0]
    theta_dot = x[:, 1]
    c_a, a, b = self.c_a, self.a, self.b
    h = 1-((theta**2)/(a**2))-((theta_dot**2)/(b**2))-(theta*theta_dot/(a*b))
    return -c_a * h

  def H(self, x):
    A = np.array([
        [1/(self.a**2), 0.5/(self.a*self.b)],
        [0.5/(self.a*self.b), 1/(self.b**2)]
    ])
    return 1 - np.einsum('bd,dh,bh->b', x, A, x)

  def get_hyp(self, x):
    if len(x.shape) == 1:
      x = np.array([x])
    return self.L_gh(x), self.neg_alpha(x) - self.L_fh(x)
  
class four_car_CBF():
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

    def H(self, x):
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
