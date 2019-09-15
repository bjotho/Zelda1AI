from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)

stop_after_a = 0
a = []
# for i in range(91):
#     a.append(3)
# for i in range(300):
#     a.append(0)
# for i in range(179):
#     a.append(4)
# for i in range(300):
#     a.append(0)
# for i in range(91):
#     a.append(3)
# for i in range(65):
#     a.append(5)

for i in range(45):
    a.append(4)
for i in range(165):
    a.append(5)
for i in range(10):
    a.append(3)
for i in range(225):
    a.append(6)
for i in range(40):
    a.append(3)
for i in range(80):
    a.append(5)

circle = [3,6,4,5]
spin_attack = [3,0,0,0,0,1,0,0,0,0,0,6,0,0,0,0,1,0,0,0,0,0,4,0,0,0,0,1,0,0,0,0,0,5,0,0,0,0,1,0,0,0,0,0]
after_kill = []
for i in range(5):
    for direction in circle:
        for j in range(10*(i+1)):
            after_kill.append(direction)
            # after_kill.append(0)
        for j in spin_attack:
            after_kill.append(j)

done = True
total_reward = 0
for step in range(3000):
    if done:
        state = env.reset()
    if step < len(a)-1:
        state, reward, done, info = env.step(a[step])
    elif stop_after_a:
        state, reward, done, info = env.step(0)
    elif info['killed_enemies'] > 0:
        index = (step-kill_step)%(len(after_kill))
        # print("killed_enemies:", info['killed_enemies'], "step:", step, "kill_step:", kill_step, "index:", index)
        state, reward, done, info = env.step(after_kill[index])
    else:
        state, reward, done, info = env.step(spin_attack[step%len(spin_attack)])
        kill_step = step
    if abs(reward) >= 15:
        print("Reward:", "%.2f" % reward, "+", "%.2f" % total_reward, "-->", "%.2f" % (total_reward + reward))
    total_reward += reward
    env.render()

env.close()
