import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(2e5)  
BATCH_SIZE = 200       
GAMMA = 0.99            
TAU = 1e-3              
LR_ACTOR = 1e-4        
LR_CRITIC = 1e-4       
WEIGHT_DECAY = 0
EPS_START = 8
EPS_EP_END = 250 
EPS_FINAL = 0 
LEARN_FOR = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, s_update = 20, num_agents = 2, seed = 0):
        
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = num_agents
        
        #Actor networks
        self.actor_local = Actor(self.state_size, self.action_size, seed = seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, seed = seed).to(device)
        
        #Critic networks
        self.critic_local = Critic(state_size*self.n_agents, action_size*self.n_agents, seed = seed).to(device)
        self.critic_target = Critic(state_size*self.n_agents, action_size*self.n_agents,seed =  seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        #optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay = WEIGHT_DECAY)
        
        #Create Noise
        self.noise = Gausian(self.action_size, seed)
        
        #create replay buffer
        self.replay_buffer = ReplayBuffer(action_size*2, BUFFER_SIZE, BATCH_SIZE, seed)
        self.softish_update_every = s_update
        
        #Set epsilon
        self.eps = EPS_START
        self.eps_decay = 1/(EPS_EP_END*s_update)
        
    def step(self, states, actions, rewards, next_states, dones, t):
        self.replay_buffer.add(states, actions, rewards, next_states, dones)
        
        if (len(self.replay_buffer.memory) > BATCH_SIZE) and (t > self.softish_update_every):
#             self.learn(self.replay_buffer.sample(), GAMMA)
            for _ in range(LEARN_FOR):
                sample = self.replay_buffer.sample()
                self.learn(sample, GAMMA)
#             self.softish_update( self.actor_local, self.actor_target, TAU)
#             self.softish_update( self.critic_local, self.critic_target, TAU)
        self.eps -= self.eps_decay
        self.eps = max(self.eps, EPS_FINAL)
        self.noise.reset()
    
    
    def act(self, states, add_noise = True):
        states = torch.from_numpy(states).float().to(device)
        if add_noise:
            noise = self.noise.sample()
        else:
            noise = 0
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).to('cpu').data.numpy()
        self.actor_local.train()
        return np.clip(action+(self.eps*noise), -1, 1)
    
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        
        states, actions, rewards, next_states, dones = experiences
        
        #critic update
        next_action1 = self.actor_target(next_states[:, :24])
        next_action2 = self.actor_target(next_states[:, 24:])
        next_actions = torch.cat((next_action1,next_action2), dim=1)
        value_next_actions = self.critic_target(next_states, next_actions)
        target_value = rewards + gamma*(value_next_actions)*(1-dones)
        expected_value = self.critic_local(states, actions)
        
        c_loss = F.mse_loss(expected_value, target_value)
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        #actor update (maximize reward)
        actors_guess1 = self.actor_local(states[:, :24])
        actors_guess2 = self.actor_local(states[:, 24:])
        actors_guesses = torch.cat((actors_guess1,actors_guess2), dim=1)
        value_estimate = self.critic_local(states, actors_guesses).mean()
        a_loss = -value_estimate
        
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        
        self.softish_update( self.actor_local, self.actor_target, TAU)
        self.softish_update( self.critic_local, self.critic_target, TAU)
    
    def softish_update(self, local, target, tau):
        
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(tau*l_param.data+(1.0-tau)*t_param.data)
        
        
        
class Gausian:
    def __init__(self, size, seed, u=0., t=0.3, s=0.5):
        self.u = u * np.ones(size)
        self.t = t
        self.s = s
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.u)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.t* (self.u - x) + self.s * np.random.standard_normal(self.u.shape)
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

