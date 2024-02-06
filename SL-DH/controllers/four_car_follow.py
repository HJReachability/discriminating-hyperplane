import numpy as np

class BasicFollow:
        def __init__(self, target, a, w):
            self.target = target
            self.a = a
            self.w = w

        def forward(self, state, t):
            x, y, theta, v = state
            x_t, y_t, theta_t, v_t = self.target
            dx = x_t - x
            dy = y_t - y
            d = np.sqrt(dx**2 + dy**2)
            # determine accel
            if v < v_t:
                accel = self.a
            elif v > v_t:
                accel = -self.a
            else:
                accel = 0

            # turning radius
            if d < 0.001:
                if v > 0:
                    accel = -self.a
                return np.array([0, accel])
            R = v / self.w
            if np.cos(theta) * dy - np.sin(theta) * dx > 0:
                max_turn_center = [x - R * np.sin(theta), y + R * np.cos(theta)]
                if ((x_t - max_turn_center[0])**2 + (y_t - max_turn_center[1])**2) < R**2:
                    accel = -self.a
                    omega = self.w
                else:
                    sin_dtheta = (np.cos(theta) * dy - np.sin(theta) * dx) / np.sqrt(dx**2 + dy**2)
                    cos_dtheta = (np.cos(theta) * dx + np.sin(theta) * dy) / np.sqrt(dx**2 + dy**2)
                    turn_radius = d / np.sqrt(1 - cos_dtheta**2)
                    if cos_dtheta < 0:
                        omega = self.w
                    else:
                        turn_radius = d / (2 * sin_dtheta)
                        omega = v / turn_radius
            else:
                max_turn_center = [x + R * np.sin(theta), y - R * np.cos(theta)]
                if (x_t - max_turn_center[0])**2 + (y_t - max_turn_center[1])**2 < R**2:  
                    accel = -self.a
                    omega = -self.w
                else:
                    sin_dtheta = (np.sin(theta) * dx - np.cos(theta) * dy) / np.sqrt(dx**2 + dy**2)
                    cos_dtheta = (np.cos(theta) * dx + np.sin(theta) * dy) / np.sqrt(dx**2 + dy**2)
                    if cos_dtheta < 0:
                        omega = -self.w
                    else:
                        turn_radius = d / (2 * sin_dtheta)
                        omega = -v / turn_radius
            return np.array([omega, accel])