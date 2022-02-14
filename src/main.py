from plot import plot_res
from Agent import Agent
import gym
import numpy as np
import sys

ENV         = 'CartPole-v1'
NB_EPISODES = 200
GAMMA       = 0.99
EPS         = 1.0
BATCH_SIZE  = 64
N_ACTIONS   = 2
EPS_END     = 0.01
INPUT_DIM   = 4
LR          = 0.0001

if __name__ == '__main__':

    render = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "-r" or sys.argv[1] == "--render":
            render = True
        else:
            print("Usage: %s [-r|--render]" % sys.argv[0])
            exit()

    env = gym.make(ENV)
    agent = Agent(gamma=GAMMA,
                  epsilon=EPS,
                  batch_size=BATCH_SIZE,
                  n_actions=N_ACTIONS,
                  eps_end=EPS_END,
                  input_dims=INPUT_DIM,
                  lr=LR)
    scores, eps_history = [], []
    n_games = NB_EPISODES

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if reward != 1:
                print(action, reward)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-30:])
        print('episode', i, '| score %.2f' % score, '| average score %.2f' % avg_score, '| epsilon %.2f' % agent.epsilon)

    plot_res(scores)

    new_env = gym.make(ENV)

    if render:
        for i in range(5):
            score = 0
            done = False
            observation = new_env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = new_env.step(action)
                score += reward
                observation = observation_
                new_env.render()
            print('Game', i, 'Score:', score)
        new_env.close()