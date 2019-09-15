from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()

    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

env.close()