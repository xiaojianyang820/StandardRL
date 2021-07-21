from .AbstractEnv import AbsEnv
import gym
from TOOLS.Logger import Logger
import numpy as np


class CartPoleEnv(AbsEnv):
    env = gym.make('CartPole-v1')
    env = env.unwrapped

    def __init__(self, logger: Logger):
        super(CartPoleEnv, self).__init__(logger=logger)

    def _obs_dim(self):
        return self.env.observation_space.shape[0]

    def _act_dim(self):
        return self.env.action_space.n

    def _obs_type(self):
        return 'Continuous'

    def _act_type(self):
        return 'Categorical'

    def _reset(self):
        obs = self.env.reset()
        return obs

    def _step(self, act):
        act_index = np.argmax(act)
        return self.env.step(act_index)

    def _render(self):
        self.env.render()

    def _close(self):
        self.env.close()


class MountainCarCateEnv(AbsEnv):
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    def __init__(self, logger: Logger):
        super(MountainCarCateEnv, self).__init__(logger=logger)

    def _obs_dim(self):
        return self.env.observation_space.shape[0]

    def _act_dim(self):
        return self.env.action_space.n

    def _obs_type(self):
        return 'Continuous'

    def _act_type(self):
        return 'Categorical'

    def _reset(self):
        obs = self.env.reset()
        return obs

    def _step(self, act):
        act_index = np.argmax(act)
        return self.env.step(act_index)

    def _render(self):
        self.env.render()

    def _close(self):
        self.env.close()


class AcrobotEnv(AbsEnv):
    env = gym.make('Acrobot-v1')
    env = env.unwrapped

    def __init__(self, logger: Logger):
        super(AcrobotEnv, self).__init__(logger=logger)

    def _obs_dim(self):
        return self.env.observation_space.shape[0]

    def _act_dim(self):
        return self.env.action_space.n

    def _obs_type(self):
        return 'Continuous'

    def _act_type(self):
        return 'Categorical'

    def _reset(self):
        obs = self.env.reset()
        return obs

    def _step(self, act):
        act_index = np.argmax(act)
        return self.env.step(act_index)

    def _render(self):
        self.env.render()

    def _close(self):
        self.env.close()


class MountainCarContiEnv(AbsEnv):
    env = gym.make('MountainCarContinuous-v0')
    env = env.unwrapped

    def __init__(self, logger: Logger):
        super(MountainCarContiEnv, self).__init__(logger)

    def _obs_dim(self):
        return self.env.observation_space.shape[0]

    def _act_dim(self):
        return self.env.action_space.shape[0]

    def _obs_type(self):
        return 'Continuous'

    def _act_type(self):
        return 'Continuous'

    def _reset(self):
        obs = self.env.reset()
        return obs

    def _step(self, act):
        obs_n, rew, done, info = self.env.step(act)
        return obs_n, rew, done, info

    def _render(self):
        self.env.render()

    def _close(self):
        self.env.close()


class PendulumEnv(AbsEnv):
    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    def __init__(self, logger: Logger):
        super(PendulumEnv, self).__init__(logger)

    def _obs_dim(self):
        return self.env.observation_space.shape[0]

    def _act_dim(self):
        return self.env.action_space.shape[0]

    def _obs_type(self):
        return 'Continuous'

    def _act_type(self):
        return 'Continuous'

    def _reset(self):
        obs = self.env.reset()
        return obs

    def _step(self, act):
        obs_n, rew, done, info = self.env.step(act)
        return obs_n, rew, done, info

    def _render(self):
        self.env.render()

    def _close(self):
        self.env.close()
