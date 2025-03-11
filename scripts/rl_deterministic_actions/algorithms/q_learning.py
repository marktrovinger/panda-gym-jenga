import numpy as np
from tqdm import tqdm

class QAgent():
    def __init__(self, state_space, action_space, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, gamma=0.95, lr=0.08):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.build_model(self.state_space, self.action_space)
        #self.replay_buffer = np.zeros([state_space, action_space])

    def build_model(self, state_space, action_space):
        Q = np.zeros((state_space, action_space))
        return Q
    
    def learn(self, env, timesteps):
        episodes = timesteps // 10
        rewards_ep = []
        successes = 0
        for episode in tqdm(range(episodes)):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            state, info = env.reset()
            #if info["is_success"] == True:
            #    #successes += 1
            #    return
            step = 0
            terminated = False
            rewards = 0
            for step in range(timesteps):   
                
                action = self.action(state)
                
                env_return = env.step(action)
                if env_return is not None: 
                    new_state, reward, terminated, _, info = env_return
                    rewards += reward
                else:
                    break
                #print(reward)
                self.Q[state, action] = self.Q[state, action] + \
                    self.lr * (reward + self.gamma*np.max(self.Q[new_state, action]) 
                            - self.Q[state, action])
                rewards_ep.append(rewards)
                if info["is_success"] == True:
                    successes += 1
                    break
                state = new_state
        print(sum(rewards_ep)/episodes)
        print(successes/episodes)

        

    def action(self, state):
        Q = self.Q[state, :]

        if np.random.rand() > self.epsilon:
            action = np.argmax(Q)
        else: 
            action = np.random.randint(self.action_space)

        return action