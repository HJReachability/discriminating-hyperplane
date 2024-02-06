import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator

class diff_game():
    def __init__(self, ser_path, grid_xlow, grid_xhigh, grid_ylow, grid_yhigh, num_points):
        with open(ser_path, 'rb') as f:

            value = pickle.load(f)

        x = np.linspace(grid_xlow, grid_xhigh, num_points)
        y = np.linspace(grid_ylow, grid_yhigh, num_points)
        self.value = value
        self.interp_f = RegularGridInterpolator((x, y), value, bounds_error=False)
    
    def forward(self, x):
        temp = self.interp_f(x)
        return temp