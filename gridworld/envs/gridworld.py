import numpy as np
import random
import gym
from gym import spaces

layers = 3

class Gridworld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.max_node = 2 ** layers - 1
        self.probs = [[.75, .25], [.1, .9]]

        self.observation_space = spaces.Box(low=np.array([1]), high=np.array([self.max_node]), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        np.random.seed(random.randint(0, 100))

        self._init_state()

    def _init_state(self):
        self.node = 1

    def _obs(self):
        return np.array([self.node - 1])

    def _reward(self):
        return np.array(self.node)

    def _done(self):
        return np.array(self.node > (self.max_node / 2))

    def step(self, a):
        '''
        0: probably left
        1: probably right
        '''

        if isinstance(a, list): a = a[0]

        p = self.probs[a]
        add = np.random.choice(2, p=p)
        self.node = self.node * 2 + add

        return self._obs(), self._reward(), self._done(), {}

    def reset(self):
        self._init_state()
        return self._obs()
