import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from infra.TruncatedNormal import TruncatedNormal
from torch.distributions.categorical import Categorical
import random


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, a=None, b=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, a, b)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a



class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, a=None, b=None):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, a=None, b=None):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
class DiscMLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.hyp = mlp([obs_dim] + list((200, 200)) + [act_dim+1], activation)
    
    # closed form
    def _distribution(self, obs, a_hyp=None, b_hyp=None):
        mu = self.mu_net(obs)
        a_dim = (mu.shape[-1])
        std = torch.exp(self.log_std)
        ab = self.hyp(obs)
        if len(obs.shape) >= 2:
            a_h, b_h = ab[:, :-1], ab[:, -1:]
        else:
            a_h, b_h = ab[:-1], ab[-1:]
        a_h = a_h.reshape(-1, a_dim) # (1, a_dim)
        b_h = b_h.reshape(-1, 1) + 1
        mu = mu.reshape(-1, a_dim)
        dot = torch.einsum('bi,bj->b', a_h, mu).reshape(-1, 1) # (b, 1)
        norm = torch.norm(mu, dim=-1, keepdim=True) # (b, 1)
        mask = dot < b_h # (b, 1)
        projection = ((dot - b_h) / norm).reshape(-1, 1)
        projection = projection * a_h
        mu = mu - mask * projection
        return Normal(mu, std)
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
class TruncMLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, a_hyp=None, b_hyp=None):
        # generates truncated normal from hyperplane constraint au>=b
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        if type(a_hyp) != type(None):
            tn_shape = (b_hyp/a_hyp).shape
            buf1 = torch.full(tn_shape, 1)
            buf2 = torch.full(tn_shape, -1)
            left = torch.zeros(tn_shape)
            right = torch.zeros(tn_shape)


            # logic switching
            left[torch.logical_and(a_hyp>=0, b_hyp>=0)] = torch.minimum(buf1, b_hyp/a_hyp)[torch.logical_and(a_hyp>=0, b_hyp>=0)]
            right[torch.logical_and(a_hyp>=0, b_hyp>=0)] = 1+1e-2

            left[torch.logical_and(a_hyp<0, b_hyp<0)] = -1-1e-2
            right[torch.logical_and(a_hyp<0, b_hyp<0)] = torch.minimum(buf1, b_hyp/a_hyp)[torch.logical_and(a_hyp<0, b_hyp<0)]

            left[torch.logical_and(a_hyp>=0, b_hyp<0)] = torch.maximum(buf2, b_hyp/a_hyp)[torch.logical_and(a_hyp>=0, b_hyp<0)]
            right[torch.logical_and(a_hyp>=0, b_hyp<0)] = 1+1e-2

            left[torch.logical_and(a_hyp<0, b_hyp>=0)] = -1-1e-2
            right[torch.logical_and(a_hyp<0, b_hyp>=0)] = torch.maximum(buf2, b_hyp/a_hyp)[torch.logical_and(a_hyp<0, b_hyp>=0)]
            mu = torch.clip(mu, left, right)
            return TruncatedNormal(mu, std, left, right)
        return Normal(mu, std)
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample().view(-1)
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
class DiscMLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = DiscMLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, a_hyp=None, b_hyp=None):
        with torch.no_grad():
            pi = self.pi._distribution(obs, a_hyp, b_hyp)
            a = pi.sample().view(-1)
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
        if type(a_hyp) == type(None):
            return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()
        else:
            return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
class TruncMLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = TruncMLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, a_hyp=None, b_hyp=None):
        with torch.no_grad():
            pi  = self.pi._distribution(obs, a_hyp, b_hyp)
            a = pi.sample().view(-1)
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
class SafeMLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_stda = -0.5 * np.ones(act_dim, dtype=np.float32)
        log_stdb = -0.5 * np.ones(1, dtype=np.float32)
        self.log_stda = torch.nn.Parameter(torch.as_tensor(log_stda))
        self.log_stdb = torch.nn.Parameter(torch.as_tensor(log_stdb))
        self.hyp = mlp([obs_dim] + list(hidden_sizes) + [act_dim+1], activation)
        #self.hyp_b = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs, a_h=None, b_h=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi_a, pi_b = self._distribution(obs)
        logp_a = None
        if a_h is not None:
            logp_a, logp_b = self._log_prob_from_distribution(pi_a, a_h, pi_b, b_h)
        return pi_a, pi_b, logp_a, logp_b

    def _distribution(self, obs):
        ab = self.hyp(obs)
        if len(obs.shape) >= 2:
            mu_a, mu_b = ab[:, :-1], ab[:, -1:]
        else:
            mu_a, mu_b = ab[:-1], ab[-1:]
        # mu_a = self.hyp_a(obs)
        # mu_b = self.hyp_b(obs)
        stda = torch.exp(self.log_stda)
        stdb = torch.exp(self.log_stdb)
        return Normal(mu_a, stda), Normal(mu_b, stdb)
    def _log_prob_from_distribution(self, pi_a, a_h, pi_b, b_h):
        return pi_a.log_prob(a_h).sum(axis=-1), pi_b.log_prob(b_h).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class SafeMLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(200,200), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = SafeMLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)
        self.action_space = action_space

    def step(self, obs):
        with torch.no_grad():
            pi_a, pi_b = self.pi._distribution(obs)
            a_h, b_h = pi_a.sample(), pi_b.sample()
            if a_h > 0:
                left, right = b_h/a_h, max(1, b_h/a_h+1)
            else:
                left, right = min(-1, b_h/a_h-1), b_h/a_h
            if left >= right:
                left, right = -1, 1
            e = random.random()
            if e > 0.01:
                pi_action = Uniform(left, right)
            else:
                pi_action = Uniform(-1, 1)
            a = pi_action.sample().view(-1)
            logp_a, logp_b = self.pi._log_prob_from_distribution(pi_a, a_h, pi_b, b_h)
            v = self.v(obs)
            vc = self.vc(obs)
        return a.numpy(), a_h.numpy(), b_h.numpy(), v.numpy(), vc.numpy(), logp_a.numpy(), logp_b.numpy()
    
    def step(self, obs):
        with torch.no_grad():
            pi_a, pi_b = self.pi._distribution(obs)
            a_h, b_h = pi_a.sample(), pi_b.sample()
            a_dim = a_h.shape[-1]
            a = torch.tensor(self.action_space.sample())
            e = random.random()
            if e > 0.05:
                # project into hyperplane
                if a_h @ a < b_h:
                    a = a - (((a_h @ a) - b_h) / torch.norm(a, dim=-1)) * a_h
            logp_a, logp_b = self.pi._log_prob_from_distribution(pi_a, a_h, pi_b, b_h)
            v = self.v(obs)
            vc = self.vc(obs)
        return a.numpy(), a_h.numpy(), b_h.numpy(), v.numpy(), vc.numpy(), logp_a.numpy(), logp_b.numpy()

    def act(self, obs):
        return self.step(obs)[0]