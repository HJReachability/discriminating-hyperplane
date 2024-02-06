import numpy as np

class target():
  def __init__(self, input_signal):
    self.input_signal = input_signal

  def forward(self, x, t):
    return self.input_signal(x, t)