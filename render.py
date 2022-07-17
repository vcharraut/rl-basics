import gym
import numpy as np
from agent import Agent

env = gym.make("Acrobot-v1")

obversation_space = env.observation_space.shape[0]

action_space = env.action_space
action_space_type = type(env.action_space).__name__

agent = Agent("reinforce", obversation_space, action_space, action_space_type,
              256, 0)
agent.load("model.pt")
for episode in range(100):
    state = env.reset()

    done = False

    while not done:
        env.render()
        action = agent.get_action(state)
        new_state, _, done, _ = env.step(action)

        state = new_state
