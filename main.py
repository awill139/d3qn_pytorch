import gym
import numpy as np
from d3qn import Agent
#plot learning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_actions = env.action_space.n
    obs_shape = list(env.observation_space.shape)
    agent = Agent(gamma = 0.99, n_actions = n_actions, epsilon = 1.0, batch_size = 64, input_dims = obs_shape)
    n_games = 500
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        # env.render()
        while not done:
            # env.render()
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, obs_, int(done))
            score += reward
            obs = obs_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[-100:])
        print('episode: {}\t curr_score: {}\t avg score: {}'.format(i, score, avg_score))

    