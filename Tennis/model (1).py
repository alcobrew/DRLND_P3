import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, layer_sizes = [200, 200, 200], seed = 0):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], action_size)   
        self.initialize_weights()
        
    def initialize_weights(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        out = torch.tanh(x)
        return out
    
    
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, layer_sizes = [200, 200, 200], seed = 0):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.fc1a = nn.Linear(state_size, layer_sizes[0])
#         self.fc1b = nn.Linear(state_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0] + action_size, layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], 1)   
        self.initialize_weights()
        
    def initialize_weights(self):
        self.fc1a.weight.data.uniform_(*hidden_init(self.fc1a))
#         self.fc1b.weight.data.uniform_(*hidden_init(self.fc1b))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        state = state.view(-1, 48)
        action = action.view(-1, 4)

        
        xa = F.leaky_relu(self.fc1a(state))
#         xb = F.relu(self.fc1b(state))
        x2 = F.leaky_relu(torch.cat([xa, action], dim=1))
        x3 = F.leaky_relu(self.fc2(x2))
        x4 = F.leaky_relu(self.fc3(x3))
        return self.fc4(x4)