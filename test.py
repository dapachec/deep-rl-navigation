# load the weights from file
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
import torch

def test(n_epi):
    agent = Agent(state_size=37, action_size=4, seed=0)
    env = UnityEnvironment(file_name="Banana.app")
    brain_name = env.brain_names[0]                    # get the default brain
    brain = env.brains[brain_name]
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
        
    for i in range(n_epi):
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        score = 0                                          # initialize the score
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            score += reward                                # update the score
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state                             # roll over the state to next time step
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_window)))

                
    env.close()

