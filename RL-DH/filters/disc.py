import numpy as np

class disc():
    def __init__(self, env, num_inputs):
        self.env = env
        self.num_inputs = num_inputs
    
    def get_hyp(self, state):
        states = np.array([state])
        inputs = self.env.sample_inputs(states, self.num_inputs)
        labels = self.env.label_inputs(states, inputs)
        a, b = self.env.label_state(inputs, labels)
        return a, b