import numpy as np
import torch
import os
import math

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Angle normalization function
def angle_normalize(x):
    # clip angle to values between -pi to pi
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = pixel_coords.reshape((-1, dim))
    # pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def get_xy_grid(xrange, yrange, nx, ny):
    xy_coords = np.stack(np.mgrid[
                         xrange[0]:xrange[1]:1j*nx, yrange[0]:yrange[1]:1j*ny
                         ], axis=-1).astype(np.float32)
    xy_coords = xy_coords.reshape((-1, 2))
    # xy_coords = torch.Tensor(xy_coords).view(-1, 2)
    return xy_coords

def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()

def to_numpy_cpu(x):
    return x.numpy()

def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()

def numerical_gradient(func, dims, eps=1e-8):
    assert (len(dims) == 1 or len(dims) == 2)
    shift_x = np.eye(dims[0])*eps
    if len(dims) == 1:
        def grad(x, t=None):
            return np.asarray([(np.array(func(x+shift_x[:, k], t)) -
                                np.array(func(x-shift_x[:, k], t))) / (2*eps) for k in range(dims[0])]).T
        return grad
    else:
        def grad_x(x, u, t=None):
            return np.asarray([(np.array(func(x+shift_x[:, k], u, t)) -
                                np.array(func(x-shift_x[:, k], u, t))) / (2*eps) for k in range(dims[0])]).T

        def grad_u(x, u, t=None):
            shift_u = np.eye(dims[1]) * eps
            return np.asarray([(np.array(func(x, u+shift_u[:, k], t)) -
                                np.array(func(x, u-shift_u[:, k], t))) / (2*eps) for k in range(dims[1])]).T

        return grad_x, grad_u

def create_helperoc_grid(grid_min, grid_max, N, pdims=None):
    grid = dict()
    grid['min'] = grid_min
    grid['max'] = grid_max
    grid['N'] = N
    grid['dim'] = grid_min.size
    dx = (grid_max - grid_min) / N
    grid['dx'] = dx
    vs = list()
    for i in range(grid['dim']):
        vs.append(np.linspace(grid_min[i], grid_max[i], N[i]))
    grid['vs'] = vs
    xs = np.meshgrid(*vs)
    grid['xs'] = xs
    grid['pdims'] = pdims
    return grid

def eval_x_at_table(x, grid):
    if isinstance(x, np.ndarray):
        if x.ndim > 2 or (x.ndim == 2 and x.shape[0] > 1 and x.shape[1] > 1):
            raise Exception("x must be (N,), (N, 1), (1, N) numpy array or list.")
        elif x.ndim == 2:
            x = x.squeeze().tolist()
        else:
            x = x.tolist()
    elif not isinstance(x, list):
        raise Exception("x must be (N,), (N, 1), (1, N) numpy array or list.")

    def _eval_idx_and_ratio_min(x_, axis_, is_periodic):
        idx = np.abs(axis_ - x_).argmin()
        if idx == 0 and x_ < axis_[0]:
            if not is_periodic:
                # TODO: THIS IS TEMPORAL FIX.
                print("WARNING: x out of grid min bound. ",
                        "Cannot evaluate value function at this point. x:%.2f bound:%.2f" % (x_, axis_[0]))
                x_ = axis_[0]
                idx_min = 0
            else:
                x_ += axis_[-1] - axis_[0]
                return _eval_idx_and_ratio_min(x_, axis_, is_periodic)
        elif idx == axis_.shape[0] - 1 and x_ > axis[-1]:
            if not is_periodic:
                # TODO: THIS IT TEMPORAL FIX
                print("WARNING: x out of grid max bound. ",
                        "Cannot evaluate value function at this point. x:%.2f bound:%.2f" % (x_, axis_[-1]))
                x_ = axis_[-1]
                idx_min = axis_.shape[0] - 2
            else:
                x_ -= axis_[-1] - axis_[0]
                return _eval_idx_and_ratio_min(x_, axis_, is_periodic)
        elif axis_[idx] <= x_ and idx < axis.shape[0] - 1:
            # index of point on axis closest to x, but smaller than x.
            idx_min = idx
        else:
            idx_min = idx - 1
        # ratio of importance of neighbor-floor point compared to neighbor-ceiling point.
        ratio_min = 1 - (x_ - axis_[idx_min]) / (axis_[1]-axis_[0])
        return idx_min, ratio_min

    indices_min = []
    ratios_min = []
    for i, axis in enumerate(grid['vs']):
        if axis.ndim > 1:
            axis = axis.squeeze()
        # print(axis)
        # idx = np.abs(axis - x[i]).argmin()
        # if idx == 0 and x[i] < axis[0]:
        #     # TODO: THIS IS TEMPORAL FIX.
        #     print("WARNING: x out of grid min bound. Cannot evaluate value function at this point.")
        #     x[i] = axis[0]
        #     idx_min = 0
        # elif idx == axis.shape[0] - 1 and x[i] > axis[-1]:
        #     if i in grid['pdims']:
        #         x[i]
        #     # TODO: THIS IT TEMPORAL FIX
        #     # print("WARNING: x out of grid max bound. ",
        #     #         "Cannot evaluate value function at this point. x:%.2f bound:%.2f" % (x[i], axis[-1]))
        #     x[i] = axis[-1]
        #     idx_min = axis.shape[0] - 2
        # elif axis[idx] <= x[i] and idx < axis.shape[0] - 1:
        #     # index of point on axis closest to x, but smaller than x.
        #     idx_min = idx
        # else:
        #     idx_min = idx - 1
        # # ratio of importance of neighbor-floor point compared to neighbor-ceiling point.
        # ratio_min = 1 - (x[i] - axis[idx_min]) / grid['dx'][i]
        idx_min, ratio_min = _eval_idx_and_ratio_min(x[i], axis, (i in grid['pdims']))
        indices_min.append(idx_min)
        ratios_min.append(ratio_min)

    indices_min = np.asarray(indices_min)
    ratios_min = np.asarray(ratios_min)
    indices_neighbor = np.vstack((indices_min, indices_min + 1))
    ratios_neighbor = np.vstack((ratios_min, 1 - ratios_min))
    return indices_neighbor, ratios_neighbor

def helper_eval_value_and_deriv(value_table, indices_neighbor, ratios_neighbor, N, do_deriv=False, dxs=None):
    weights_value = np.zeros(2 ** N, dtype=np.float64)
    vertices_value = np.zeros(2 ** N, dtype=np.float64)
    vertices_derivs = np.zeros((N, 2**N), dtype=np.float64)
    for j in range(2 ** N):
        bin_indices_tail = [int(b) for b in bin(j)[2:]]
        tail_length = len(bin_indices_tail)
        if tail_length < N:
            bin_indices_head = (N - tail_length) * [0]
            bin_indices = bin_indices_head + bin_indices_tail
        else:
            bin_indices = bin_indices_tail

        indices = []
        weight = 1.
        for dim, bin_index in enumerate(bin_indices):
            indices.append(indices_neighbor[bin_index, dim])
            weight = weight * ratios_neighbor[bin_index, dim]

        vertices_value[j] = value_table[tuple(indices)]
        weights_value[j] = weight
        # if do_deriv, get deriv for each vertices of indices_neighbor.
        if do_deriv:
            for i in range(N):
                if indices[i] == 0:
                    # Trapezoidal rule.
                    indices_up = deepcopy(indices)
                    indices_up[i] = indices_up[i] + 1
                    vertices_derivs[i, j] = (value_table[tuple(indices_up)]
                        - value_table[tuple(indices)]) / dxs[i]
                elif indices[i] == value_table.shape[i]-1:
                    indices_down = deepcopy(indices)
                    indices_down[i] = indices_down[i] - 1
                    vertices_derivs[i, j] = (value_table[tuple(indices)]
                        - value_table[tuple(indices_down)]) / dxs[i]
                else:
                    # Trapezoidal rule.
                    indices_up = deepcopy(indices)
                    indices_up[i] = indices_up[i] + 1
                    indices_down = deepcopy(indices)
                    indices_down[i] = indices_down[i] - 1
                    vertices_derivs[i, j] = .5 * (value_table[tuple(indices_up)]
                        - value_table[tuple(indices_down)]) / dxs[i]

    value = np.dot(vertices_value, weights_value)
    if do_deriv:
        deriv = np.matmul(vertices_derivs, weights_value)
    else:
        deriv = None
    return value, deriv

def eval_value_from_table(x, value_table, grid):
    """ x: query state (list of length n).
        grid: list with length n. each grid[i] corresponds to i-th axis grid intervals.
        dxs: list with length n. dxs indicates grid interval length of each i-th axis.
    """
    indices_neighbor, ratios_neighbor = eval_x_at_table(x, grid)
    value, _ = helper_eval_value_and_deriv(
        value_table, indices_neighbor, ratios_neighbor, grid['dim'])
    return value

def eval_deriv_from_table(x, value_table, grid):
    """ x: query state (list of length n).
        grid: list with length n. each grid[i] corresponds to i-th axis grid intervals.
        dxs: list with length n. dxs indicates grid interval length of each i-th axis.
    """
    indices_neighbor, ratios_neighbor = eval_x_at_table(x, grid)
    _, deriv = _helper_eval_value_and_deriv(
        value_table, indices_neighbor, ratios_neighbor, grid['N'], True, grid['dx'])
    return deriv