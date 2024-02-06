from envs.four_car import four_car
from controllers.constant import target
from filters.CBF_filters import four_car_CBF
from filters.NN_filters import FCN
from filters.disc import disc
from inv_set.four_car import cbf
from controllers.QP import QP_CBF_controller
from controllers.four_car_follow import BasicFollow
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import time
import matplotlib.animation as animation


# initialize environment and cbf
kv = 1
kw = 1
ka = 1
mu = 0
se = 0
w = 2
a = 1
r = 2
gamma = 1
sys_dt = 0.01
all_disc_dt = [0.1, 0.5, 1]
all_eps = [0.3, 0.3, 0.3]

alpha = [0.6, 1.0, 1.0]

cbf_inv_set = cbf(a, r, gamma)
env = four_car(kv, kw, ka, mu, se, w, a, sys_dt, 0, cbf_inv_set)

target_state = np.array([6, -3, 0, 1])
target_controller = BasicFollow(target_state, a, w)
cbf_filter = four_car_CBF(a, r, gamma)
cbf_controller = QP_CBF_controller(target_controller, cbf_filter, 2)

x = np.array([-6, -2, 1, 0.5])
t = 15

y_target, u_target, H_target = env.run_system(x, t, target_controller)
y_cbf, u_cbf, H_cbf = env.run_system(x, t+5, cbf_controller)

fig1, ax1 = plt.subplots() 
circle1 = plt.Circle((0, 0), r, color='black', label='obstacle')
ax1.add_patch(circle1)
ax1.set_xlim([-8, 8])
ax1.set_ylim([-8, 8])
ax1.plot(y_target[:, 0], y_target[:, 1], label='u_ref')
ax1.plot(y_cbf[:, 0], y_cbf[:, 1], label='CBF-QP')
ax1.scatter(-6, -2, c='r', s=25, label='Start Position')
ax1.scatter(6, -3, marker='*', s=25, c='r', label='Goal')

fig, ax = plt.subplots() 
axins = zoomed_inset_axes(ax, 2, loc=9) # zoom = 2
ax.plot(range(len(H_target)), H_target, label='u_ref', linewidth=1)
axins.plot(range(len(H_target)), H_target, linewidth=1)
ax.plot(range(len(H_cbf)), H_cbf, label='CBF-QP', linewidth=1)
axins.plot(range(len(H_cbf)), H_cbf, linewidth=1)
ax.plot(range(len(H_target)), np.zeros(len(H_target)), label='h(x)=0', c='r', linewidth=1)
axins.plot(range(len(H_target)), np.zeros(len(H_target)), c='r', linewidth=1)


for disc_dt, eps, alpha in zip(all_disc_dt, all_eps, alpha):

    # training params and init model
    num_states = 8000
    num_inputs = 500
    epochs = 100
    num_log = 4
    ratio = 0.0 # percentage of safe states

    # model params
    input_size = 5
    output_size = 3
    num_layers = 3
    size = 2000
    activation = 'relu'
    lr = 1e-4
    sc = 0.99
    gamma_pos = 1
    gamma_neg = 5

    def preproc(states):
        x, y, theta, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        sin = np.sin(theta)
        cos = np.cos(theta)
        states = np.column_stack((x, y, sin, cos, v))
        return states

    model = FCN(input_size, output_size, num_layers, size, activation, lr, sc, gamma_pos, gamma_neg, preproc)
    checkpoint = th.load(f'exps/four_car/model_dt{disc_dt}.pth')
    model.FCN.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.optimizer.param_groups[0]['lr'] = lr
    model.FCN.eval()
                
    nn_controller = QP_CBF_controller(target_controller, model, 2, eps=eps)
    y_nn, u_nn, H_nn = env.run_system(x, t+5, nn_controller)

    print("Logging NN traj")


    # plot trajectory results
    #plt.plot(y_disc[:, 0], y_disc[:, 1], label='disc')
    if disc_dt == 0.5:
        style = '--'
    else:
        style = '-'
    ax1.plot(y_nn[:, 0], y_nn[:, 1], style,  c='g', label=f'NN-QP ($\Delta t_L={disc_dt}$)', linewidth=1.5, alpha=alpha)

    # plot cbf
    #plt.plot(range(len(H_disc)), H_disc, label='disc')
    ax.plot(range(len(H_nn)), H_nn, style, c='g', label=f'NN-QP ($\Delta t_L={disc_dt}$)', linewidth=1.5, alpha=alpha,)
    axins.plot(range(len(H_nn)), H_nn, style, c='g', alpha=alpha, linewidth=2)

ax1.grid()
ax1.set_xlabel('x position')
ax1.set_ylabel('y position')
fig1.savefig('exps/four_car/traj_dubins_dt.png')




axins.set_xlim(385, 535)
axins.set_ylim(-0.1, 0.8)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="black")
ax.grid()
ax.set_xlabel('timestep')
ax.set_ylabel('h(x)')
fig.savefig('exps/four_car/cbf_dubins_dt_zoom.png')


def anim():


    fig, ax2 = plt.subplots()
    ax2.set_xlim([-8, 8])
    ax2.set_ylim([-8, 8])
    circle1 = plt.Circle((0, 0), r, color='black', label='obstacle')
    ax2.add_patch(circle1)
    line1, = ax2.plot([], [])
    line2, = ax2.plot([], [], '--')
    line3, = ax2.plot([], [], c='grey', linewidth=0.5)
    line4, = ax2.plot([], [], c='grey', linewidth=1)
    ax2.scatter(-6, -2, c='r', s=25, label='Start Position')
    ax2.scatter(6, -3, marker='*', s=25, c='r', label='Goal')

    def h(self, x):
        px, py, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return (px + (v**2)/(4*self.a)*np.cos(theta))**2 + (py + (v**2)/(4*self.a)*np.sin(theta))**2

    def update(num, line1, line2):
        # plot traj
        line1.set_data(y_nn[:num, 0], y_nn[:num, 1])
        x = y_nn[num]

        # plot cbf
        px, py, theta_car, v = x[0], x[1], x[2], x[3]
        a = (v**2)/(4)*np.cos(theta_car)
        b = (v**2)/(4)*np.sin(theta_car)
        theta = np.linspace(0, 2*np.pi, 100)
        x = -a + (r + v**2/(4))*np.cos(theta)  # Calculate x-coordinates
        y = -b + (r + v**2/(4))*np.sin(theta)  # Calculate y-coordinates
        line2.set_data(x, y)

        # plot car
        r_car = 0.2
        x_car = px + r_car * np.cos(theta)
        y_car = py + r_car * np.sin(theta)
        line3.set_data(x_car, y_car)

        # plot heading
        r_car = 0.05
        head_x_car = np.cos(theta_car)*r_car + px + r_car * np.cos(theta)
        head_y_car = np.sin(theta_car)*r_car + py + r_car * np.sin(theta)
        line4.set_data(head_x_car, head_y_car)
        return [line1,line2,line3,line4]

    anim = animation.FuncAnimation(fig, update, len(y_nn), fargs=[line1, line2],
                    interval=20, blit=True)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')


    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('exps/four_car/traj_anim.gif')

anim()