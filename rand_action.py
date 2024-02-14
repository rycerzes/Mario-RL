import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

ENV_NAME = 'SuperMarioBros-1-1-v3'
DISPLAY = True
NUM_OF_EPISODES = 50_000


env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    while not done:
        a = env.action_space.sample()  # Take random action
        new_state, reward, done, truncated, info  = env.step(a)
        state = new_state
env.close()
