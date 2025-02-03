def main():
    pass

def run_episode(env,agent,verbose = 1):

    s = env.reset()
    agent.reset_memory()
    max_step = env.n_stops
    episode_reward = 0
    
    i = 0
    while i < max_step:

        # Remember the states
        #agent.remember_state(s)

        # Choose an action
        a = agent.action(s)
        
        # Take the action, and get the reward from environment
        s_next,r,done = env.step(a)

        # Tweak the reward
        r = -1 * r
        
        if verbose: print(s_next,r,done)
        
        # Update our knowledge in the Q-table
        agent.train(s,a,r,s_next)
        
        # Update the caches
        episode_reward += r
        s = s_next
        
        # If the episode is terminated
        i += 1
        if done:
            break
            
    return env,agent,episode_reward

if __name__ == "__main__":
    main()