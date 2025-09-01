import gymnasium as gym
try:
    gym.make('LunarLander-v3')
    print('LunarLander-v3 found.')
except gym.error.NameNotFound:
    print('LunarLander-v3 not found, trying LunarLander-v2.')
    try:
        gym.make('LunarLander-v2')
        print('LunarLander-v2 found.')
    except gym.error.NameNotFound:
        print('Neither LunarLander-v3 nor LunarLander-v2 found.')