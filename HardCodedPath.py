from nes_py.wrappers import JoypadSpace
import gym_zelda_1
from gym_zelda_1.actions import MOVEMENT

env = gym_zelda_1.make('Zelda1-v0')
env = JoypadSpace(env, MOVEMENT)

stop_after_a = 1
a = []
for i in range(30):
    a.append(5)     # up
for i in range(45):
    a.append(4)     # left
for i in range(150):
    a.append(5)     # up
for i in range(70):
    a.append(4)     # left
for i in range(35):
    a.append(5)     # up
for i in range(30):
    a.append(4)     # left

# Go to level 1 from beginning.
# for i in range(45):
#     a.append(4)     # left
# for i in range(165):
#     a.append(5)     # up
# for i in range(10):
#     a.append(3)     # right
# for i in range(225):
#     a.append(6)     # down
# for i in range(40):
#     a.append(3)     # right
# for i in range(75):
#     a.append(5)     # up
# for i in range(120):
#     a.append(3)     # right
# for i in range(50):
#     a.append(5)     # up
# for i in range(80):
#     a.append(3)     # right
# for i in range(210):
#     a.append(5)     # up
# for i in range(15):
#     a.append(4)     # left
# for i in range(120):
#     a.append(5)     # up
# for i in range(200):
#     a.append(4)     # left
# for i in range(20):
#     a.append(5)     # up

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
memory_testing_last = 119
for step in range(10000):
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
    # if info['memory_testing'] != memory_testing_last:
    print("memory testing:", info['memory_testing'], "(", info['x_pos'], ",", info['y_pos'], ")")
    # memory_testing_last = info['memory_testing']
    # if abs(reward) >= 15:
    #     print("Reward:", "%.2f" % reward, "+", "%.2f" % total_reward, "-->", "%.2f" % (total_reward + reward))
    total_reward += reward
    env.render()

env.close()
