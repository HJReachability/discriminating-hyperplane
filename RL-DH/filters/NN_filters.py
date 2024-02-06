import infra.pytorch_utils as ptu
import numpy as np
import torch as th
from torch import nn
from torch import optim

class FCN():
  def __init__(self, input_size, output_size, n_layers, size, activation, lr, sc, gamma_pos, gamma_neg, preproc=lambda x:x):
    '''
    n_layers: number of hidden layers
    Size: dim of hidden layers
    '''
    self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    self.FCN = ptu.build_mlp(input_size, output_size, n_layers, size, activation).to(self.device)
    self.loss_a = nn.MSELoss()
    self.optimizer = optim.Adam(self.FCN.parameters(), lr)
    self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=sc)
    self.gamma_pos = gamma_pos
    self.gamma_neg = gamma_neg
    self.preproc = preproc

  def forward(self, states):
    states = th.tensor(states).to(self.device).to(th.float32)
    if len(states.shape) == 1:
      states = states.unsqueeze(0)
    ab = self.FCN(states)
    return ab[:, :-1], ab[:, -1]

  def update2(self, states, inputs, input_labels, env):
    states = self.preproc(states)
    inputs = th.tensor(inputs).to(self.device)
    a, b = self.forward(states)
    a = th.unsqueeze(a, 2).double()
    u_hat = th.bmm(inputs, a).squeeze(2) - b.unsqueeze(1)
    u_bar = th.tensor(input_labels).to(self.device)
    num_inputs = inputs.shape[1]

    # format diff and mask for positive labels
    diff_pos = -u_bar*u_hat*th.where(u_bar > 0, 1, 0)
    #diff_pos = diff_pos / th.abs(diff_pos + 1e-12)
    mask_pos = th.where(diff_pos >= 0, 1, 0)
    num_pos = th.sum(u_bar > 0, axis=1) + 1e-12

    # format diff and mask for negative labels
    diff_neg = -u_bar*u_hat*th.where(u_bar < 0, 1, 0)
    #diff_neg = diff_neg / th.abs(diff_neg + 1e-12)
    mask_neg = th.where(diff_neg >= 0, 1, 0)
    num_neg = th.sum(u_bar < 0, axis=1) + 1e-12

    

    # calculate combined loss
    self.optimizer.zero_grad()
    loss = (self.gamma_pos*th.mean(diff_pos*mask_pos, dim=1) + self.gamma_neg*th.mean(diff_neg*mask_neg, dim=1)).mean(axis=0)
    loss.backward()
    self.optimizer.step()
    return loss.item()
  

  def get_val_loss(self, states, inputs, input_labels, env):
      states = self.preproc(states)
      self.FCN.eval()
      with th.no_grad():
        inputs = th.tensor(inputs).to(self.device)
        a, b = self.forward(states)
        a = th.unsqueeze(a, 2).double()
        u_hat = th.bmm(inputs, a).squeeze(2) - b.unsqueeze(1)
        u_bar = th.tensor(input_labels).to(self.device)
        num_inputs = inputs.shape[1]
        
        diff_pos = -u_bar*u_hat*th.where(u_bar > 0, 1, 0)
        mask_pos = th.where(diff_pos >= 0, 1, 0)
        num_pos = th.sum(u_bar > 0, axis=1) + 1e-12

        diff_neg = -u_bar*u_hat*th.where(u_bar < 0, 1, 0)
        mask_neg = th.where(diff_neg >= 0, 1, 0)
        num_neg = th.sum(u_bar < 0, axis=1) + 1e-12
        loss = (self.gamma_pos*th.mean(diff_pos*mask_pos, dim=1) + self.gamma_neg*th.mean(diff_neg*mask_neg, dim=1)).mean(axis=0)
      self.FCN.train()
      return loss.item()



  def get_hyp(self, states):
    a, b = self.forward(states)
    return a.detach(), b.detach()