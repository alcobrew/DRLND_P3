Model success: For this project I leveraged an actor critic DDPG method from former lesson and modified it to complete the tennis project. I ran into several issues initially with learning, as the agent struggled to produce consistent gains in it's average score. Once I implemented a "learn for" function, a function to randomize the initial weights, I included the former actions in the second layer of my model, and I decreased the size of the overall model, it started to learn quite smoothly. The hyper parameters for the agent were as follows: BUFFER_SIZE = int(1e5)
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
The model uses an Actor critic DDPG format with 2 networks, an actor and a critic, each with 4 layers. The Actor layer sizes were L1 = 200, L2 = 200, L3 = 200, 4a = action_size. The Critic Layer sizes were L1 = 200 + Action Size, L2 = 200, L3 = 200, 4a = 1. The models used relu activation funtions except the Actor output function was a Tanh function. In order to improve the agents learning performance further, I'd look into fine tuning the replay buffer by using a prioritized replay experience. By storing the TD error with every experience in the replay buffer, we can then sample the most valuable experiences at a higher rate and make sure that they aren't overwritten when the replay buffer fills up
