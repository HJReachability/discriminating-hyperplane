from envs.MooreGreitzerJet import jet
from inv_set.jet import diff_game
from filters.NN_filters import FCN
from controllers.QP import QP_CBF_controller
import numpy as np
import matplotlib.pyplot as plt
from controllers.constant import target
from filters.disc import disc
import torch as th

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

        with th.no_grad():
                a, b = model.forward(eval_state)
                a = a.item()
                b = b.item()
                ax.axvline(b/a, 0, c='r', label='network')

        ax.scatter(eval_input[i2], np.ones_like(eval_input[i2]), s=2, label='safe')
        ax.scatter(eval_input[i1], -1* np.ones_like(eval_input[i1]), s=2, label='unsafe')
        #ax.plot(u1, u2_disc, 'g', label="svm fit")
        ax.set_xlim((-0.5, 0.5))
    
    if train:
        fig.savefig('exps/jet/jet_inputs_train.png')
    else:
        fig.savefig('exps/jet/jet_inputs_val.png')
    plt.clf()

    plt.plot(range(len(train_losses)), train_losses, label='train loss')
    plt.plot(range(len(val_losses)), val_losses, label='val losss')
    plt.legend()
    plt.savefig('exps/jet//loss.png')
    plt.clf()
    plt.close('all')

grid_xlow = -0.75
grid_xhigh = 0.75
grid_ylow = -0.75
grid_yhigh = 0.75
num_points = 400
path_to_diff_game = 'inv_set/jet_is_ser.pkl'
inv_set = diff_game(path_to_diff_game, grid_xlow, grid_xhigh, grid_ylow, grid_yhigh, num_points)

# initialize system
u_max = 0.5
d_max = 0.0
sys_dt = 0.005
disc_dt = 0.025

env = jet(u_max, d_max, sys_dt, disc_dt, inv_set)

# initialize network
input_size = 2
output_size = 2
n_layers = 5
size = 1000
activation = 'relu'
lr = 5e-4
sc = 1
gamma_pos = 1
gamma_neg = 10
model = FCN(input_size, output_size, n_layers, size, activation, lr, sc, gamma_pos, gamma_neg)
model.FCN.train()

# train network
num_states = 15000
num_inputs = 300
epochs = 700 # 700 epochs
num_log = 4
eps = 0.1

train_losses = []
val_losses = []
best_loss = float('inf')
# val_states, val_inputs, val_labels = env.sample_data(num_states, num_inputs, ratio)
# print(len(val_states))
val_states, val_inputs, val_labels = env.sample_data(num_states, num_inputs)
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
            }, 'exps/jet/model.pth')
        train_losses = train_losses[-100:]
        val_losses = val_losses[-100:]
        print("epoch:", epoch, 'train loss:', train_loss)
        print("epoch:", epoch, 'val loss:', val_loss)
        v_uns_idx = np.any(val_labels < 0, axis=1)
        t_uns_idx = np.any(train_labels < 0, axis=1)
        log(env, val_states[v_uns_idx][:num_log], val_inputs[v_uns_idx][:num_log], val_labels[v_uns_idx][:num_log], model, train_losses, val_losses, False)
        log(env, train_states[t_uns_idx][:num_log], train_inputs[t_uns_idx][:num_log], train_labels[t_uns_idx][:num_log], model, train_losses, val_losses, True)

model.FCN.eval()

checkpoint = th.load('exps/jet/model.pth')
model.FCN.load_state_dict(checkpoint['model_state_dict'])
model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.optimizer.param_groups[0]['lr'] = lr

# reference controller
def bang(x, t):
    return np.array([0])

ref = target(bang)

# nn controller
nn_controller = QP_CBF_controller(ref, model, 1, eps=eps)

# SVM controller
svm_filter = disc(env, 20000)
svm_controller = QP_CBF_controller(ref, svm_filter, 1, eps=0.1)
    

start_state = np.array([-0.05, 0.45])
t = 15
traj, u_hist, h_hist = env.run_system(start_state, t, ref)
traj_nn, u_hist_nn, h_hist_nn = env.run_system(start_state, t, nn_controller)
traj_svm, u_hist_svm, h_hist_svm = env.run_system(start_state, t, svm_controller)

# grid info for plotting
# ~~~~~~~~~~~~~~
x_grid = np.linspace(grid_xlow, grid_xhigh, num_points)
y_grid = np.linspace(grid_ylow, grid_yhigh, num_points)
X, Y = np.meshgrid(x_grid, y_grid)
# ~~~~~~~~~~~~~~


plt.contour(X, Y, inv_set.value.T, levels=[0], colors='black', linestyles='dashed')
plt.scatter(start_state[0], start_state[1], c='r', s=10)
plt.plot(traj[:, 0], traj[:, 1], c='blue', label="ref")
plt.plot(traj_nn[:, 0], traj_nn[:, 1], c='red', label="NN-QP")
plt.plot(traj_svm[:, 0], traj_svm[:, 1], c='green', label="SVM-QP")
plt.legend()
plt.xlim(grid_xlow, grid_xhigh)
plt.ylim(grid_ylow, grid_yhigh)
plt.savefig('exps/jet/traj_jet.png')
plt.clf()

# plt.plot(range(len(h_hist)), h_hist)
# plt.savefig('exps/jet/h_hist.png')
# plt.clf()

# plt.plot(range(len(u_hist)), u_hist)
# plt.savefig('exps/jet/u_hist.png')
# plt.clf()

plt.contour(X, Y, inv_set.value.T, levels=[0], colors='black', linestyles='dashed')

theta = np.linspace(0, 2 * np.pi, 100)  # Angle values
r = 0.5  # Radius of the circle
x = r * np.cos(theta)  # x-coordinates
y = r * np.sin(theta)  # y-coordinates

# Plot the circle as a dashed red line
# plt.plot(x, y, 'r--')
# states = env.sample_states(500)
# for state in states:
#     traj, u_hist, h_hist = env.run_system(state, t, nn_controller)
#     if np.any(h_hist > 0):
#         plt.plot(traj[:, 0], traj[:, 1], c='red', linewidth=0.1)
#     else:
#         plt.plot(traj[:, 0], traj[:, 1], c='blue', linewidth=0.1)
#     plt.scatter(state[0], state[1], c='r', s=0.5)

#     plt.xlim(-0.5, 0.5)
#     plt.ylim(-0.5, 0.5)
#     plt.savefig('exps/jet/traj_jet.png')