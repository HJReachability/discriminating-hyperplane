from envs.inv_ped import inv_ped
from controllers.constant import target
from controllers.QP import QP_CBF_controller
from filters.CBF_filters import inv_ped_CBF
from filters.disc import disc
from filters.NN_filters import FCN
from inv_set.inv_ped import cbf
import numpy as np
import matplotlib.pyplot as plt
import torch as th
plt.rcParams.update({'font.size': 12})

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
        a_cbf, b_cbf = cbf_filter.get_hyp(eval_state)
        ax.axvline(b_cbf/a_cbf, 0, c='g', label='cbf')

        ax.scatter(eval_input[i2], np.ones_like(eval_input[i2]), s=2, label='safe')
        ax.scatter(eval_input[i1], -1* np.ones_like(eval_input[i1]), s=2, label='unsafe')
        #ax.plot(u1, u2_disc, 'g', label="svm fit")
        ax.set_xlim((-3, 3))
    
    if train:
        fig.savefig('exps/inv_ped/inv_ped_inputs_train.png')
    else:
        fig.savefig('exps/inv_ped/inv_ped_inputs_val.png')
    plt.clf()

    plt.plot(range(len(train_losses)), train_losses, label='train loss')
    plt.plot(range(len(val_losses)), val_losses, label='val losss')
    plt.legend()
    plt.savefig('exps/inv_ped//loss.png')
    plt.clf()
    plt.close('all')

# initialize inerted pendulum
c_a, a, b, m, g, l = 0.8, 0.075, 0.15, 2, 10, 1
cbf_filter = inv_ped_CBF(c_a, a, b, m, g, l)
u_min, u_max = -3, 3
sys_dt = 0.005
disc_dt = 0.05
h_decay = 0.99
cbf_inv_set = cbf(c_a, a, b, m, g, l)
env = inv_ped(g, m, l, u_min, u_max, sys_dt, disc_dt, cbf_inv_set)

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
num_states = 10000
num_inputs = 300
epochs = 400 # 400 epochs
num_log = 4
eps = 0.01

train_losses = []
val_losses = []
best_loss = float('inf')
# val_states, val_inputs, val_labels = env.sample_data(num_states, num_inputs, ratio)
# print(len(val_states))
val_states, val_inputs, val_labels = env.sample_data(num_states, num_inputs)
plt.scatter(val_states[:, 0], val_states[:, 1])
plt.savefig('exps/inv_ped/states.png')
plt.clf()
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
            }, 'exps/inv_ped/model.pth')
        train_losses = train_losses[-10:]
        val_losses = val_losses[-10:]
        print("epoch:", epoch, 'train loss:', train_loss)
        print("epoch:", epoch, 'val loss:', val_loss)
        log(env, val_states[:num_log], val_inputs[:num_log], val_labels[:num_log], model, train_losses, val_losses, False)
        log(env, train_states[:num_log], train_inputs[:num_log], train_labels[:num_log], model, train_losses, val_losses, True)

model.FCN.eval()

# reload network
checkpoint = th.load('exps/inv_ped/model.pth')
model.FCN.load_state_dict(checkpoint['model_state_dict'])
model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.optimizer.param_groups[0]['lr'] = lr

# define controllers
class inv_ped_target():
  # target input for jason paper
  def __init__(self):
      pass
  
  def forward(x, t):
    if t < 2:
      return 3
    elif t < 4:
      return -3
    elif t < 6:
      return 3
    else:
      return m*(l**2)*((-g/l)*np.sin(x[0]) - (1.5*x[0]+1.5*x[1]))
  
target_controller = target(inv_ped_target)

cbf_filter = inv_ped_CBF(c_a, a, b, m, g, l)
cbf_controller = QP_CBF_controller(inv_ped_target, cbf_filter, 1)

disc_filter = disc(env, 10000)
disc_controller = QP_CBF_controller(inv_ped_target, disc_filter, 1)

nn_controller = QP_CBF_controller(inv_ped_target, model, 1, eps=eps)

# run system and log results
print("Running exps...")
x = np.array([-0.01, 0])
t = 15
traj_cbf, u_hist_cbf, h_hist_cbf = env.run_system(x, t, cbf_controller)
# traj_disc, u_hist_disc, h_hist_disc = env.run_system(x, t, disc_controller)
# traj_disc2, u_hist_disc2, h_hist_disc2 = env_old.run_system(x, t, disc_controller_old)
traj_nn, u_hist_nn, h_hist_nn = env.run_system(x, t, nn_controller)

n = 50
def f(x, y):
    x = np.stack((x, y), axis=2).reshape(-1, 2)
    A = np.array([
        [1/(0.075**2), 0.5/(0.075*0.15)],
        [0.5/(0.075*0.15), 1/(0.15**2)]
    ])
    return (1 - np.einsum('bd,dh,bh->b', x, A, x)).reshape(n, n)

plt.figure(figsize=(8,6))

x = np.linspace(-0.1, 0.1, n)
y = np.linspace(-0.2, 0.2, n)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z, colors='black', levels=0, linestyles='dashed', label='control invariant set')

plt.xlim(-0.1, 0.1)
plt.ylim(-0.2, 0.2)
plt.plot(traj_cbf[:, 0], traj_cbf[:, 1], label='CBF-QP', c='r')
# plt.plot(traj_disc[:, 0], traj_disc[:, 1], label='disc')
plt.plot(traj_nn[:, 0], traj_nn[:, 1], label='NN-QP', c='g')
plt.legend()
plt.xlabel('$\\theta$')
plt.ylabel('$\dot{\\theta}$')
plt.grid()
plt.savefig('exps/inv_ped/traj_inv_ped.png', bbox_inches="tight")
plt.clf()

plt.figure(figsize=(8,6))
plt.ylim(-3, 3)
plt.plot(range(len(u_hist_cbf)), u_hist_cbf, label='CBF-QP', c='r')
# plt.plot(range(len(u_hist_disc)), u_hist_disc, label='disc h_decay=0.99')
# plt.plot(range(len(u_hist_disc2)), u_hist_disc2, label='disc h_decay=0')
plt.plot(range(len(u_hist_nn)), u_hist_nn, label='NN-QP', c='g')
plt.legend()
plt.xlabel('timestep')
plt.ylabel('u')
plt.grid()
plt.savefig('exps/inv_ped/u_inv_ped.png')
plt.clf()

plt.ylim(-1, 2)
plt.plot(range(len(h_hist_cbf)), h_hist_cbf, label='CBF-QP')
# plt.plot(range(len(h_hist_disc)), h_hist_disc, label='disc h_decay=0.99')
# plt.plot(range(len(h_hist_disc2)), h_hist_disc2, label='disc h_decay=0')
plt.plot(range(len(h_hist_nn)), h_hist_nn, label='NN-QP')
plt.legend()
plt.savefig('exps/inv_ped/h_h_decay.png')
plt.clf()