import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_block(input_dim, output_dim, activation=nn.Tanh, initialize=True, normalize=True):
    layer = nn.Linear(input_dim, output_dim)
    if initialize:
        layer = layer_init(layer)

    layers = [layer]
    if normalize:
        layers.append(nn.LayerNorm(output_dim))
    layers.append(activation())

    return layers

def mlp(input_dim, output_dim, hidden_dims, norm=True, init=True, output_activation=nn.Tanh):
    layers = layer_block(input_dim, hidden_dims[0], initialize=init, normalize=norm)
    for i in range(len(hidden_dims) - 1):
        layers.extend(layer_block(hidden_dims[i], hidden_dims[i + 1], initialize=init, normalize=norm))
    layers.extend(layer_block(hidden_dims[-1], output_dim, activation=output_activation, initialize=init, normalize=norm))

    return nn.Sequential(*layers)

class ActorCritic(nn.Module):
    def __init__(self, env_config):
        super().__init__()
        self.state_dim = env_config["state_dim"]
        self.action_dim = env_config["action_dim"]
        self.num_discretes = env_config["num_discretes"]
        self.is_continuous = env_config["is_continuous"]

        ###################### Implement here : 1. Neural Network ########################
        self.critic = mlp(self.state_dim, 1, [128, 128], output_activation=nn.Identity)
        if self.is_continuous:
            self.actor_mean = mlp(self.state_dim, self.action_dim, [64, 64], norm=False)
            self.actor_logstd = nn.Parameter(0.5 * torch.zeros(1, self.action_dim))

        else:
            self.actor_logit = mlp(self.state_dim, self.action_dim, [64, 64], norm=False, output_activation=nn.Softmax)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        ###################### Implement here : policy distribution ########################
        if self.is_continuous:
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)  # Use torch distribution Noraml
            if action is None:
                action = probs.rsample()
            return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)
        else:
            logits = self.actor_logit(x)
            probs = Categorical(logits)  # Use torch distribution Categorical
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)