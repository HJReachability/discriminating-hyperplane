from envs.cartpole import Cartpole
from controllers.constant import target
from controllers.QP import QP_CBF_controller
from filters.disc import disc
from inv_set.cartpole import trap
from filters.NN_filters import FCN
from inv_set.inv_ped import cbf
import numpy as np
import matplotlib.pyplot as plt
import torch as th

# initialize cartpole
inv_set = trap(0.5, -5)
disc_steps = 2
env = Cartpole(inv_set, disc_steps)

# initialize network
input_size = 4
output_size = 2
n_layers = 5
size=1000
activation = 'relu'
lr = 5e-4
sc = 1
gamma_pos = 1
gamma_neg = 5
model = FCN(input_size, output_size, n_layers, size, activation, lr, sc, gamma_pos, gamma_neg)
model.FCN.train()

# train network
num_states = 10000
num_inputs = 300
epochs = 200
num_log = 4

train_losses = []
val_losses = []
best_loss = float('inf')
val_states, val_inputs, val_labels = env.sample_data(num_states, num_inputs)
train_states, train_inputs, train_labels = env.sample_data(num_states, num_inputs)
plt.scatter(val_states[:, 0], val_states[:, 1])
plt.savefig('exps/cartpole/states.png')
plt.clf()
print("Training model...")
for epoch in range(epochs):
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
            }, 'exps/cartpole/model.pth')
        train_losses = train_losses[-500:]
        val_losses = val_losses[-500:]
        print("epoch:", epoch, 'train loss:', train_loss)
        print("epoch:", epoch, 'val loss:', val_loss)

model.FCN.eval()

# reload network

# define controllers
class bang():
    def __init__(self):
        pass

    def forward(x, t):
        if t <= 10:
            return 1
        elif t <= 20:
            return -1
        else:
            return 1
    
  
target_controller = bang()

disc_filter = disc(env, 1000)
disc_controller = QP_CBF_controller(bang, disc_filter, 1)

nn_controller = QP_CBF_controller(bang, model, 1)

# run system and log results
print("Running exps...")
state = env.reset()
t = 30
traj_disc = env.run_system(state, t, disc_controller)
traj_nn = env.run_system(state, t, nn_controller)
plt.scatter(state[0], state[1], c='r')

plt.xlim(-1, 1)
plt.ylim(-6, 6)
plt.plot(traj_disc[:, 0], traj_disc[:, 1], label='disc')
plt.plot(traj_nn[:, 0], traj_nn[:, 1], label='nn')
plt.legend()
plt.savefig('exps/cartpole/traj.png')
plt.clf()

# plt.ylim(-3, 3)
# plt.plot(range(len(u_hist_disc)), u_hist_disc, label='disc')
# plt.plot(range(len(u_hist_nn)), u_hist_nn, label='nn')
# plt.legend()
# plt.savefig('exps/inv_ped/u.png')
# plt.clf()

# plt.ylim(-1, 2)
# plt.plot(range(len(h_hist_disc)), h_hist_disc, label='disc')
# plt.plot(range(len(h_hist_nn)), h_hist_nn, label='nn')
# plt.legend()
# plt.savefig('exps/inv_ped/h.png')
# plt.clf()