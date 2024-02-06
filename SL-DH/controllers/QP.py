import cvxpy as cp
import numpy as np

class QP_CBF_controller():
  def __init__(self, target, filter, input_dim, eps=0):
    self.target = target
    self.filter = filter
    self.input_dim = input_dim
    self.eps=eps

  def forward(self, x, t):
    a,b = self.filter.get_hyp(x)
    a_norm = np.linalg.norm(a)
    a /= a_norm
    b /= a_norm
    target_u = self.target.forward(x, t)
    P = np.eye(self.input_dim)
    u = cp.Variable(self.input_dim)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(u-target_u, P)),
                      [a@u >= b + self.eps]
                    )
    prob.solve()
    r = u.value
    return r