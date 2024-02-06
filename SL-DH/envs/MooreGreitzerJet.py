from scipy.integrate import odeint
from sklearn.svm import LinearSVC
import numpy as np

class jet():
    def __init__(self, u_max, d_max, sys_dt, disc_dt, inv_set):
        self.u_max = u_max
        self.d_max = d_max
        self.sys_dt = sys_dt
        self.disc_dt = disc_dt
        self.inv_set = inv_set

    def run_system(self, x_0, num_steps, controller):
        traj = [x_0]
        h_hist = []
        u_hist = []
        x_t = x_0
        step = 0
        while step < num_steps:
            steps = np.linspace(step, step+self.sys_dt)
            u = controller.forward(x_t, step)[0]
            d = np.random.uniform(-self.d_max, self.d_max, 1)[0]
            traj_delta_t = np.array(odeint(self.system, x_t, steps, args=(u,d,))[-1])
            traj.append(traj_delta_t)
            step += self.sys_dt
            x_t = traj_delta_t
            u_hist.append(u)
            h_hist.append(self.inv_set.forward(np.array([x_t])))
        return np.array(traj), np.array(u_hist), np.array(h_hist)
    
    def system(self, state, t, u, d):
        x = state[0]
        y = state[1]
        return np.array([-y-(3/2)*(x**2)-(1/2)*(x**3)+d,
                         (0.8076+u)*x-0.9424*y])

    def next_states(self, states, inputs, dists):
        num_states, num_inputs, input_dim = inputs.shape
        x = states[:, 0]
        y = states[:, 1]
        dists_ext = np.insert(dists, 1, 0, axis=1)
        x_dot = -y-(3/2)*(x**2)-(1/2)*(x**3)
        y_dot = 0.8076*x-0.9424*y
        state_dot = np.hstack((x_dot.reshape(-1, 1), y_dot.reshape(-1, 1)))
        state_dot = state_dot + dists_ext
        state_dot.reshape(num_states, 1, -1)
        inputs = np.insert(inputs*x.reshape(-1, 1, 1), 0, 0, axis=2)
        state_dot = state_dot.reshape(-1, 1, 2) + inputs
        states_next = states.reshape(num_states, 1, -1) + self.disc_dt * state_dot
        return states_next

    def label_inputs(self, states, inputs, dists=0):
        num_states, num_inputs, input_sim = inputs.shape
        if type(dists) == int:
            dists = np.zeros((num_states, 1))
        x_next = self.next_states(states, inputs, dists)
        labels = -np.sign(self.inv_set.forward(x_next.reshape(-1, 2))).reshape(num_states, num_inputs)
        return labels
    
    def label_state(self, inputs, input_labels):
        if np.all(input_labels == 1.0):
            a = np.array([1.0])
            b = np.array([-0.5])
        elif np.all(input_labels == -1.0):
            a = np.array([1.0])
            b = np.array([0.5])
        else:
            lsvc = LinearSVC(verbose=0, class_weight={-1:1, 1:1})
            lsvc.fit(inputs.reshape(-1, 1), np.ravel(input_labels.reshape(-1, 1)))
            a=lsvc.coef_[0]
            b=-lsvc.intercept_
        return a, b

    def sample_inputs(self, states, num_inputs):
        num_states = states.shape[0]
        inputs = np.random.uniform(-self.u_max, self.u_max, (num_states, num_inputs, 1))
        return inputs
    
    def sample_dists(self, states):
        num_states = states.shape[0]
        dists = np.random.uniform(-self.d_max, self.d_max, (num_states, 1))
        return dists

    def sample_states(self, num_states):
        x = np.random.uniform(-0.51, 0.51, (num_states, 1))
        y = np.random.uniform(-0.51, 0.51, (num_states, 1))
        states = np.hstack((x, y))
        return states[self.inv_set.forward(states) <= 0]

    def sample_data(self, num_states, num_inputs):
        states = self.sample_states(num_states)
        while len(states) <= num_states:
            states = np.vstack((states, self.sample_states(num_states)))
        inputs = self.sample_inputs(states, num_inputs)
        dists = self.sample_dists(states)
        labels = self.label_inputs(states, inputs, dists)
        return states[:num_states], inputs[:num_states], labels[:num_states]