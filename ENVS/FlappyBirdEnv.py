from .flappy_bird.wrapped_flappy_bird import GameState
from .AbstractEnv import AbsImageEnv
import numpy as np
import cv2


class FlappyBirdEnv(AbsImageEnv):

    frame_size = (80, 80)

    def _obs_dim(self):
        # 目前环境对象向策略网络返回的状态矩阵为连续4帧80X80的二值型图片
        obs_dim = (self.frame_size[0], self.frame_size[1], 4)
        return obs_dim

    def _act_dim(self):
        act_dim = 2
        return act_dim

    def _obs_type(self):
        return 'Continuous'

    def _act_type(self):
        return 'Categorical'

    def _img_transform(self, singal_frame):
        singal_frame = cv2.cvtColor(cv2.resize(singal_frame, self.frame_size), cv2.COLOR_BGR2GRAY)
        _, singal_frame = cv2.threshold(singal_frame, 1, 255, cv2.THRESH_BINARY)
        return singal_frame

    def _reset(self):
        try:
            del self.env
        except Exception as e:
            pass
        self.env = GameState()
        init_act = np.array([1, 0])
        singal_frame_obs_n, _, _ = self.env.frame_step(init_act)
        singal_frame_obs_n = self._img_transform(singal_frame_obs_n)
        self.obs_t = (singal_frame_obs_n, singal_frame_obs_n, singal_frame_obs_n, singal_frame_obs_n)
        return np.stack(self.obs_t, axis=2)

    def _step(self, act):
        singal_frame_obs_n, rew_t, don_n = self.env.frame_step(act)
        singal_frame_obs_n = self._img_transform(singal_frame_obs_n)
        self.obs_t = (singal_frame_obs_n, ) + self.obs_t[:3]
        return np.stack(self.obs_t, axis=2), rew_t, don_n, {}

    def _render(self):
        pass

    def _close(self):
        pass
