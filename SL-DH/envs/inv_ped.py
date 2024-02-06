from scipy.integrate import odeint
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np

class inv_ped():
    def __init__(self, g, m, l, u_min, u_max, sys_dt, disc_dt, inv_set):
        self.u_min = u_min
        self.u_max = u_max
        self.g = g
        self.m = m
        self.l = l
        self.sys_dt = sys_dt
        self.disc_dt = disc_dt
        self.inv_set = inv_set

    def run_system(self, x_0, num_steps, controller):
        """From start state, num_steps, and controller, return trajectory"""
        traj = [x_0]
        u_hist = []
        x_t = x_0
        step = 0
        h_hist = []
        while step < num_steps:
            steps = np.linspace(step, step+self.sys_dt)
            u = controller.forward(x_t, step)
            traj_delta_t = np.array(odeint(self.system, x_t, steps, args=(u,))[-1])
            traj.append(traj_delta_t)
            step += self.sys_dt
            x_t = traj_delta_t
            u_hist.append(u)
            h_hist.append(self.inv_set.forward(np.array([x_t])))
        return np.array(traj), np.array(u_hist), np.array(h_hist)
    
    def system(self, x, t, u):
        """Sys dynamics"""
        g = self.g
        m = self.m
        l = self.l
        theta = x[0]
        theta_dot = x[1]
        f = np.array([theta_dot, (g/l)*np.sin(theta)])
        g = np.array([0, 1/(m*l**2)])
        return f+g*u
    
    def next_states(self, x, inputs):
        '''Based on dt discretization, estimate next states from inputs
        x: (b, d)
        inputs: (b, i, h)
        out: (b, i, d)
        '''
        num_states, num_inputs, input_dim = inputs.shape
        g = self.g
        m = self.m
        l = self.l
        theta = x[:, 0]
        theta_dot = x[:, 1]
        f = np.zeros_like(x)
        f[:, 0] = theta_dot
        f[:, 1] = (g/l)*np.sin(theta)
        g = np.array([0, 1/(m*l**2)])
        gu = g.reshape(1, 1, -1) * inputs
        x_dot = f.reshape(num_states, 1, -1) + gu
        x_next = x.reshape(num_states, 1, -1) + self.disc_dt * x_dot
        return x_next
    
    def label_inputs(self, x, inputs):
        num_states, num_inputs, input_dim = inputs.shape
        x_next = self.next_states(x, inputs)
        labels = np.sign(self.inv_set.forward(x_next.reshape(-1, 2))).reshape(num_states, num_inputs)
        return labels
    
    def label_state(self, inputs, input_labels):
        if np.all(input_labels == 1.0):
            a = np.array([1])
            b = np.array([-3])
        elif np.all(input_labels == -1.0):
            a = np.array([1])
            b = np.array([3])
        else: 
            lsvc = LinearSVC(verbose=0, class_weight={-1:1, 1:1})
            lsvc.fit(inputs.reshape(-1, 1), np.ravel(input_labels.reshape(-1, 1)))
            a=lsvc.coef_[0]
            b=-lsvc.intercept_
        return a, b

    def sample_inputs(self, states, num_inputs):
        num_states = states.shape[0]
        inputs = np.random.uniform(-3, 3, (num_states, num_inputs, 1))
        inputs[:, 0] = -3
        inputs[:, -1] = 3
        return inputs
    
    def sample_states(self, num_states):
        theta, theta_dot = np.random.uniform(-0.3, 0.3, num_states), np.random.uniform(-0.6, 0.6, num_states)
        states = np.column_stack((theta, theta_dot))
        states = states[self.inv_set.forward(states) > 0]
        return states
    
    def sample_data(self, num_states, num_inputs):
        states = self.sample_states(num_states)
        while len(states) <= num_states:
            states = np.vstack((states, self.sample_states(num_states)))
        inputs = self.sample_inputs(states, num_inputs)
        labels = self.label_inputs(states, inputs)
        return states[:num_states], inputs[:num_states], labels[:num_states]