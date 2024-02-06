from envs.four_car import four_car
from controllers.constant import target
from filters.CBF_filters import four_car_CBF
from filters.NN_filters import FCN
from filters.disc import disc
from inv_set.four_car import cbf
from controllers.QP import QP_CBF_controller
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import time


def log(env, eval_states, eval_inputs, eval_labels, model, train_losses, val_losses, train):
    fig, axs = plt.subplots(len(eval_inputs))
    for i in range(len(axs)):
        ax = axs[i]
        eval_input = eval_inputs[i]
        eval_label = eval_labels[i]
        eval_state = eval_states[i]
        i1 = (eval_label == -1)
        i2 = (eval_label == 1)
        # svm_a, svm_b = env.label_state(eval_input, eval_label)
        # svm_a1, svm_a2 = svm_a
        # u1 = np.linspace(-2, 2, 100)
        # u2_disc = (svm_b - svm_a1*u1) / svm_a2
        h = env.inv_set.forward(np.array([eval_state]))

        with th.no_grad():
                a, b = model.get_hyp(eval_state)
                a1, a2 = a[0]
                u1 = np.linspace(-2, 2, 100)
                u2_model = (b - a1*u1) / a2
                ax.plot(u1, u2_model, 'r', label='network')

        ax.scatter(eval_input[i2, 0], eval_input[i2, 1], s=0.75, label='safe')
        ax.scatter(eval_input[i1, 0], eval_input[i1, 1], s=0.75, label='unsafe')
        #ax.plot(u1, u2_disc, 'g', label="svm fit")
        ax.set_box_aspect(1)
        ax.set_title(h)
        plt.subplots_adjust(hspace=0)
        ax.set_xlim((-2, 2))
        ax.set_ylim((-1, 1))
    
    if train:
        fig.savefig('exps/four_car/4car_inputs_train.png')
    else:
        fig.savefig('exps/four_car/4car_inputs_val.png')
    plt.clf()

    plt.plot(range(len(train_losses)), train_losses, label='train loss')
    plt.plot(range(len(val_losses)), val_losses, label='val losss')
    plt.legend()
    plt.savefig('exps/four_car/loss.png')
    plt.clf()
    plt.close('all')

def log_eval(env, traj, controller):
    for x in traj[125:175]:
        state = np.array([x])
        u = controller.forward(x, 0)
        inputs = env.sample_inputs(state, 1000)
        batch_safe, batch_unsafe = env.label_inputs(state, inputs)
        safe_states, safe_inputs, safe_labels = batch_safe
        unsafe_states, unsafe_inputs, unsafe_labels = batch_unsafe
        labels = np.vstack((safe_labels, unsafe_labels))
        i1 = (labels == -1)
        i2 = (labels == 1)
        with th.no_grad():
                a, b = model.forward(state)
                a1, a2 = a[0]
                a1, a2 = a1.item(), a2.item()
                b = b.item()
                u1 = np.linspace(-2, 2, 100)
                u2_model = (b - a1*u1) / a2
                plt.plot(u1, u2_model, 'r', label='network')
        plt.scatter(u[0], u[1], c='g', s=3, label='QP input')
        plt.scatter(inputs[i2, 0], inputs[i2, 1], s=0.75, label='safe')
        plt.scatter(inputs[i1, 0], inputs[i1, 1], s=0.75, label='unsafe')
        plt.subplots_adjust(hspace=0)
        plt.legend()
        plt.xlim((-2, 2))
        plt.ylim((-1, 1))
        plt.title(str(x))
        plt.savefig('exps/four_car/eval_input.png')
        plt.clf()
        time.sleep(0.2)


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
sys_dt = 0.05
disc_dt = 0.5
eps = 0.3

cbf_inv_set = cbf(a, r, gamma)
env = four_car(kv, kw, ka, mu, se, w, a, sys_dt, disc_dt, cbf_inv_set)

# training params and init model
num_states = 8000
num_inputs = 500
epochs = 100 #100 epochs
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

print("Init network...")
model = FCN(input_size, output_size, num_layers, size, activation, lr, sc, gamma_pos, gamma_neg, preproc)
model.FCN.train()

train_losses = []
val_losses = []
best_loss = float('inf')
val_states, val_inputs, val_labels = env.sample_data(num_states, num_inputs)
# print(len(val_states))
print("Training model...")
for epoch in range(epochs):
    train_states, train_inputs, train_labels = env.sample_data(num_states, num_inputs)
    for i in range(5):
        train_loss = model.update2(train_states, train_inputs, train_labels, env)
    model.scheduler.step()
    if epoch % 1 == 0:
        train_losses.append(train_loss)
        val_loss = model.get_val_loss(val_states, val_inputs, val_labels, env)
        val_losses.append(val_loss)
        if val_loss < best_loss:
            th.save({
            'model_state_dict': model.FCN.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            }, f'exps/four_car/model_dt{disc_dt}.pth')
        train_losses = train_losses[-50:]
        val_losses = val_losses[-50:]
        print("epoch:", epoch, 'train loss:', train_loss)
        print("epoch:", epoch, 'val loss:', val_loss)
        log(env, val_states[:num_log], val_inputs[:num_log], val_labels[:num_log], model, train_losses, val_losses, False)
        log(env, train_states[:num_log], train_inputs[:num_log], train_labels[:num_log], model, train_losses, val_losses, True)

model.FCN.eval()

checkpoint = th.load(f'exps/four_car/model_dt{disc_dt}.pth')
model.FCN.load_state_dict(checkpoint['model_state_dict'])
model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.optimizer.param_groups[0]['lr'] = lr

print("Exps...")
# define controllers
class four_car_target():
  def __init__(self):
      pass
  
  def forward(self, state, t):
    return np.array([-1, 0.5])
    #return np.array([t*np.sin(t), np.cos(t)])

class BasicFollow:
    def __init__(self, target):
        self.target = target

    def forward(self, state, t):
        x, y, theta, v = state
        x_t, y_t, theta_t, v_t = self.target
        dx = x_t - x
        dy = y_t - y
        d = np.sqrt(dx**2 + dy**2)
        # determine accel
        if v < v_t:
            accel = a
        elif v > v_t:
            accel = -a
        else:
            accel = 0

        # turning radius
        if d < 0.001:
            if v > 0:
                accel = -a
            return np.array([0, accel])
        R = v / w
        if np.cos(theta) * dy - np.sin(theta) * dx > 0:
            max_turn_center = [x - R * np.sin(theta), y + R * np.cos(theta)]
            if ((x_t - max_turn_center[0])**2 + (y_t - max_turn_center[1])**2) < R**2:
                accel = -a
                omega = w
            else:
                sin_dtheta = (np.cos(theta) * dy - np.sin(theta) * dx) / np.sqrt(dx**2 + dy**2)
                cos_dtheta = (np.cos(theta) * dx + np.sin(theta) * dy) / np.sqrt(dx**2 + dy**2)
                turn_radius = d / np.sqrt(1 - cos_dtheta**2)
                if cos_dtheta < 0:
                    omega = w
                else:
                    turn_radius = d / (2 * sin_dtheta)
                    omega = v / turn_radius
        else:
            max_turn_center = [x + R * np.sin(theta), y - R * np.cos(theta)]
            if (x_t - max_turn_center[0])**2 + (y_t - max_turn_center[1])**2 < R**2:  
                accel = -a
                omega = -w
            else:
                sin_dtheta = (np.sin(theta) * dx - np.cos(theta) * dy) / np.sqrt(dx**2 + dy**2)
                cos_dtheta = (np.cos(theta) * dx + np.sin(theta) * dy) / np.sqrt(dx**2 + dy**2)
                if cos_dtheta < 0:
                    omega = -w
                else:
                    turn_radius = d / (2 * sin_dtheta)
                    omega = -v / turn_radius
        return np.array([omega, accel])
            

target_state = np.array([6, -3, 0, 1])
target_controller = BasicFollow(target_state)
#target_controller = four_car_target()
cbf_filter = four_car_CBF(a, r, gamma)
disc_inputs = 10000
disc_filter = disc(env, disc_inputs)
cbf_controller = QP_CBF_controller(target_controller, cbf_filter, 2)
disc_controller = QP_CBF_controller(target_controller, disc_filter, 2)
nn_controller = QP_CBF_controller(target_controller, model, 2, eps=eps)

# test system and plot results

x = np.array([-6, -2, 1, 0.5])
t = 15

# target input
y_target, u_target, H_target = env.run_system(x, t, target_controller)

# truth cbf
y_cbf, u_cbf, H_cbf = env.run_system(x, t+5, cbf_controller)

# # discretized svm controller
#y_disc, u_disc, H_disc = env.run_system(x, t, disc_controller)

# # trained nn controller
y_nn, u_nn, H_nn = env.run_system(x, t+5, nn_controller)

print("Logging NN traj")
#log_eval(env, y_nn, nn_controller)

theta = np.linspace(0, 2*np.pi, 100)
x = r*np.cos(theta)
y = r*np.sin(theta)
plt.plot(x, y, color='r', label='obstacle')


fig, ax = plt.subplots() 
circle1 = plt.Circle((0, 0), r, color='black', label='obstacle')
ax.add_patch(circle1)
# plot trajectory results
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])
ax.plot(y_target[:, 0], y_target[:, 1], label='u_ref')
ax.plot(y_cbf[:, 0], y_cbf[:, 1], label='CBF-QP')
#plt.plot(y_disc[:, 0], y_disc[:, 1], label='disc')
ax.plot(y_nn[:, 0], y_nn[:, 1], label=f'NN-QP ($\Delta t_l={disc_dt}$)')
ax.scatter(-6, -2, c='r', s=25, label='Start Position')
ax.scatter(6, -3, marker='*', s=25, c='r', label='Goal')
ax.legend()
ax.grid()
ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title('Trajectory of Dubin\'s car with filtered point-follow controller')
fig.savefig('exps/four_car/traj_dubins.png')
fig.clf()

# # plot input
# plt.plot(range(u_target.shape[0]), u_target[:, 0], label='target')
# plt.plot(y_cbf[:, 0], y_cbf[:, 1], label='cbf')
# #plt.plot(range(u_disc.shape[0]), u_disc[:, 0], label='disc')
# plt.plot(y_nn[:, 0], y_nn[:, 1], label='nn')
# plt.legend()
# plt.savefig('exps/four_car/u_0.png')
# plt.clf()

# # plot input
# plt.plot(range(u_target.shape[0]), u_target[:, 1], label='target')
# plt.plot(y_cbf[:, 0], y_cbf[:, 1], label='cbf')
# #plt.plot(range(u_disc.shape[0]), u_disc[:, 1], label='disc')
# plt.plot(y_nn[:, 0], y_nn[:, 1], label='nn')
# plt.legend()
# plt.savefig('exps/four_car/u_1.png')
# plt.clf()

# # # plot heading
# plt.plot(np.linspace(0, t, len(y_target[:, 2])), y_target[:, 2], label='target')
# plt.plot(np.linspace(0, t, len(y_target[:, 2])), y_cbf[:, 2], label='cbf')
# #plt.plot(np.linspace(0, t, len(y_target[:, 2])), y_disc[:, 2], label='disc')
# plt.plot(np.linspace(0, t, len(y_target[:, 2])), y_nn[:, 2], label='nn')
# plt.legend()
# plt.savefig('exps/four_car/theta.png')
# plt.clf()

# # # plot velocity
# plt.plot(np.linspace(0, t, len(y_target[:, 3])), y_target[:, 3], label='u_ref')
# plt.plot(np.linspace(0, t, len(y_target[:, 3])), y_cbf[:, 3], label='CBF-QP')
# #plt.plot(np.linspace(0, t, len(y_target[:, 3])), y_disc[:, 3], label='disc')
# plt.plot(np.linspace(0, t, len(y_target[:, 3])), y_nn[:, 3], label='NN-QP')
# plt.legend()
# plt.savefig('exps/four_car/vel.png')
# plt.clf()

# plot cbf
fig, ax = plt.subplots() 
ax.plot(range(len(H_target)), H_target, label='u_ref')
ax.plot(range(len(H_cbf)), H_cbf, label='CBF-QP')
#plt.plot(range(len(H_disc)), H_disc, label='disc')
ax.plot(range(len(H_nn)), H_nn, label='NN-QP')
ax.plot(range(len(H_target)), np.zeros(len(H_target)), label='h(x)=0', linewidth=1)
ax.grid()
ax.legend()
ax.set_xlabel('timestep')
ax.set_ylabel('h(x)')
ax.set_title('h(x) over time')
fig.savefig('exps/four_car/cbf_dubins.png')
fig.clf()
plt.clf()
