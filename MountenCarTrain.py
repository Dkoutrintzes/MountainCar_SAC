import gym
import os
import time
import pygame
import matplotlib.pyplot as plt
import torch
from Sac_Network.Discrete_SAC_Agent import SACAgent

print(torch.cuda.is_available())
env = gym.make('MountainCar-v0')
sac_agent = SACAgent(env)

print('Observetion Space',env.observation_space)
print('Action Space', env.action_space)
#pygame.init()

TIME_LIMIT = 500
env = gym.wrappers.TimeLimit(
    gym.envs.classic_control.MountainCarEnv(),
    max_episode_steps=TIME_LIMIT + 1,
)

last_score = -500
run_score = -500
best = -500
for episode  in range(400):
    last_score = run_score
    if best < last_score:
        best = last_score
    run_score = 0
    state = env.reset()
    done = False
    evaluation_episode = episode % 20 == 1
    print('\r', f'Episode: {episode + 1}/{200} | Last Score: {last_score} | Best: {best}', end=' ')
    
    for step in range(500):
        #print(step)
        if evaluation_episode:
        
            action = sac_agent.get_next_action(state, evaluation_episode=True)

            next_state,reward,done,info = env.step(action.item())
            
            run_score += reward
            #print(reward)
            #print(step,done)
            if done:
                break
            state = next_state
            env.render(mode='human')
            time.sleep(0.01)
            #print(reward)
        else:
            #print(step)
        
            action = sac_agent.get_next_action(state, evaluation_episode=False )
            #print(action)
            next_state,reward,done,info = env.step(action)
            sac_agent.train_on_transition(state, action, next_state, reward, done)
            run_score += reward
            if done:
                break
            state = next_state
            #env.render(mode='human')
            time.sleep(0.01)
            #print(reward)
    # if evaluation_episode:
    #     print('Episode Reward: ',run_score)

# for episode  in range(20):
#     run_score = 0
#     state = env.reset()
#     done = False
#     evaluation_episode =  True
#     for step in range(1000):
#         action = sac_agent.get_next_action(state, evaluation_episode=evaluation_episode)

#         next_state,reward,done,info = env.step(action)
        
#         run_score += reward
#         if done:
#             break
#         state = next_state
#         env.render(mode='human')
#         time.sleep(0.01)
#         #print(reward)
    
#     print('Episode Reward: ',run_score)


env.close()