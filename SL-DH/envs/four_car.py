from scipy.integrate import odeint
from sklearn import svm
import numpy as np

class four_car():
    def __init__(self, kv, kw, ka, mu, se, w, a, sys_dt, disc_dt, inv_set):
        self.kv = kv
        self.kw = kw
        self.ka = ka
        self.mu = mu
        self.se = se
        self.w = w
        self.a = a
        self.sys_dt = sys_dt
        self.disc_dt = disc_dt
        self.inv_set = inv_set
        self.max_v = 5
        self.min_v = 0

    def run_system(self, x_0, total_time, input_signal):
        y = [x_0]
        u_total = []
        H_total = []
        a_hist = []
        b_hist = []
        x_t = x_0
        t = 0
        while t <= total_time:
            x_t[2] = x_t[2] % (2*np.pi)
            timesteps = np.linspace(t, t+self.sys_dt)
            u = input_signal.forward(x_t, t)
            if x_t[3] < self.min_v:
                x_t[3] = 0
                u[1] = max(0, u[1])
            elif x_t[3] > self.max_v:
                x_t[3] = self.max_v
                u[1] = min(0, u[1])
            u_total.append(u)
            cbf_val = self.inv_set.forward(np.array([x_t]))
            H_total.append(cbf_val)
            #print(x_t, cbf_val)
            y_delta_t = np.array(odeint(self.system, x_t, timesteps, args=(u,))[-1])
            y.append(y_delta_t)
            t += self.sys_dt
            x_t = y_delta_t
        return np.array(y), np.array(u_total), np.array(H_total)

    def system(self, x, t, u):
        if len(x.shape) == 1:
            x = np.array([x])
        f = self.f(x)
        g = self.g(x)
        if len((f + u@g).shape) == 0:
            print(f, u, g)
        return (f + u@g).reshape(4)

    def f(self, x):
        kv = self.kv
        mu = self.mu
        se = self.se
        px, py, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        f0 = kv*v*np.cos(theta)
        f1 = kv*v*np.sin(theta)
        f2 = np.zeros(x.shape[0])
        f3 = -mu*v+se*self.h(px, py)
        return np.column_stack((f0, f1, f2, f3))
    
    def g(self, x):
        kw = self.kw
        ka = self.ka
        g = np.array([
            [0, 0],
            [0, 0],
            [kw, 0],
            [0, ka]
        ])
        return g.T

    def h(self, px, py):
        return (px**2 + py**2)**0.1

    def next_states(self, x, u):
        g_x = self.g(x)
        f_x = np.expand_dims(self.f(x), axis=1)
        x = np.expand_dims(x, axis=1)
        return x + self.disc_dt*(f_x + u@g_x)

    def sample_states(self, num_states):
        x, y = np.random.uniform(-5, 5, num_states), np.random.uniform(-5, 5, num_states)
        theta = np.random.uniform(0, 2*np.pi, num_states)
        v = np.random.uniform(0, 5, num_states)
        states = np.column_stack((x, y, theta, v))
        states = states[np.logical_and(self.inv_set.forward(states) >= 0, self.inv_set.forward(states) <= 2)]
        return states

    def sample_inputs(self, states, num_inputs):
        num_states = states.shape[0]
        w = np.random.uniform(-self.w, self.w, num_states*num_inputs)
        a = np.random.uniform(-self.a, self.a, num_states*num_inputs)
        inputs = np.column_stack((w, a)).reshape((num_states, num_inputs, 2))
        return inputs

    def label_inputs(self, states, inputs):
        num_states, num_inputs, input_dim = inputs.shape
        x_next = self.next_states(states, inputs)
        labels = np.sign(self.inv_set.forward(x_next.reshape(-1, 4))).reshape(num_states, num_inputs)
        return labels

    # def label_inputs(self, states, inputs):
    #     next_states = self.next_states(states, inputs)
    #     num_states, num_inputs = states.shape[0], inputs.shape[1]
    #     labels = np.sign(self.inv_set.forward(next_states.reshape((-1, 4)))).reshape(num_states, num_inputs)
    #     unsafe_mask = np.any(labels < 0, axis=1)
    #     safe_mask = np.all(labels >= 0, axis=1)
    #     return (states[safe_mask], inputs[safe_mask], labels[safe_mask]), (states[unsafe_mask], inputs[unsafe_mask], labels[unsafe_mask])

    def label_state(self, inputs, input_labels):
        inputs = inputs.reshape(-1, 2)
        input_labels = input_labels.reshape(-1)
        if np.all(input_labels == 1.0):
            a = [1, 0]
            b = -2
        elif np.all(input_labels == -1.0):
            a = [1, 0]
            b = 2
        else: 
            lsvc = svm.SVC(kernel='linear', C = 1.0, class_weight={-1:100, 1:1})
            lsvc.fit(inputs, input_labels)
            a=lsvc.coef_[0]
            b=-lsvc.intercept_
        return a, b

    # def sample_data(self, num_states, num_inputs, ratio):
    #     num_safe, num_unsafe = int(ratio*num_states), int((1-ratio)*num_states)
    #     states = self.sample_states(num_states)
    #     inputs = self.sample_inputs(states, num_inputs)
    #     batch_safe, batch_unsafe = self.label_inputs(states, inputs)
    #     safe_states, safe_inputs, safe_labels = batch_safe
    #     unsafe_states, unsafe_inputs, unsafe_labels = batch_unsafe
    #     while len(safe_states) < num_safe or len(unsafe_states) < num_unsafe:
    #         states = self.sample_states(num_states)
    #         inputs = self.sample_inputs(states, num_inputs)
    #         batch_safe, batch_unsafe = self.label_inputs(states, inputs)
    #         batch_safe_states, batch_safe_inputs, batch_safe_labels = batch_safe
    #         batch_unsafe_states, batch_unsafe_inputs, batch_unsafe_labels = batch_unsafe
    #         if len(safe_states) < num_safe:
    #             safe_states = np.vstack((safe_states, batch_safe_states))
    #             safe_inputs = np.vstack((safe_inputs, batch_safe_inputs))
    #             safe_labels = np.vstack((safe_labels, batch_safe_labels))
    #         if len(unsafe_states) < num_unsafe:
    #             unsafe_states = np.vstack((unsafe_states, batch_unsafe_states))
    #             unsafe_inputs = np.vstack((unsafe_inputs, batch_unsafe_inputs))
    #             unsafe_labels = np.vstack((unsafe_labels, batch_unsafe_labels))
    #     safe_states, unsafe_states = safe_states[:num_safe], unsafe_states[:num_unsafe]
    #     safe_inputs, unsafe_inputs = safe_inputs[:num_safe], unsafe_inputs[:num_unsafe]
    #     safe_labels, unsafe_labels = safe_labels[:num_safe], unsafe_labels[:num_unsafe]
    #     states = np.vstack((safe_states, unsafe_states))
    #     inputs = np.vstack((safe_inputs, unsafe_inputs))
    #     labels = np.vstack((safe_labels, unsafe_labels))
    #     idx = np.arange(len(states))
    #     np.random.shuffle(idx)
    #     return states[idx], inputs[idx], labels[idx]

    # def sample_data(self, num_states, num_inputs, ratio):
    #     states = self.sample_states(num_states)
    #     inputs = self.sample_inputs(num_states, num_inputs)
    #     safe, unsafe = self.label_inputs(states, inputs)
    #     print(safe[0].shape, unsafe[0].shape)
    #     safe_states, safe_inputs, safe_labels = safe
    #     unsafe_states, unsafe_inputs, unsafe_labels = unsafe
    #     num_unsafe = len(unsafe_states)
    #     num_safe = int(num_unsafe/ratio) - num_unsafe
    #     safe_states, unsafe_states = safe_states[:num_safe], unsafe_states[:num_unsafe]
    #     safe_inputs, unsafe_inputs = safe_inputs[:num_safe], unsafe_inputs[:num_unsafe]
    #     safe_labels, unsafe_labels = safe_labels[:num_safe], unsafe_labels[:num_unsafe]
    #     states = np.vstack((safe_states, unsafe_states))
    #     inputs = np.vstack((safe_inputs, unsafe_inputs))
    #     labels = np.vstack((safe_labels, unsafe_labels))
    #     idx = np.arange(len(states))
    #     np.random.shuffle(idx)
    #     return states[idx], inputs[idx], labels[idx]

    def sample_data(self, num_states, num_inputs):
        states = self.sample_states(num_states)
        inputs = self.sample_inputs(states, num_inputs)
        labels = self.label_inputs(states, inputs)
        idx = np.any(labels < 0, axis=1)
        states = states[idx]
        inputs = inputs[idx]
        labels = labels[idx]
        while len(states) <= num_states:
            batch_states = self.sample_states(num_states)
            batch_inputs = self.sample_inputs(batch_states, num_inputs)
            batch_labels = self.label_inputs(batch_states, batch_inputs)
            idx = np.any(batch_labels < 0, axis=1)
            batch_states = batch_states[idx]
            batch_inputs = batch_inputs[idx]
            batch_labels = batch_labels[idx]
            states = np.vstack((states, batch_states))
            inputs = np.vstack((inputs, batch_inputs))
            labels = np.vstack((labels, batch_labels))
        return states[:num_states], inputs[:num_states], labels[:num_states]
        
