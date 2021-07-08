import numpy as np

class OrnsteinUhlenbeckActionNoise(object):
    """
    奥恩斯坦-乌伦贝克随机过程：该随机过程由两个相对独立的过程来组合而成，第一个是回归过程，也就是随机状态向量向
    均值向量回归；第二个过程是维纳过程，也就是高斯噪音的积分结果，它倾向于保持当前的随机状态向量不变
    """
    def __init__(self, mu: np.ndarray, sigma: float, theta: float = 0.15, dt: float = 0.01, x0: np.ndarray = None):
        """
        :param mu: np.ndarray,
            随机过程的回归均值向量（一维）
        :param sigma: float,
            维纳过程中蕴含的高斯噪音的标准差
        :param theta: float,
            随机过程中向均值回归的力度系数
        :param dt: float,
            回合控制中每一个回合的间隔时间
        :param x0: np.ndarray,
            整个随机过程的初始状态向量（一维）
        """
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        """
        重置状态向量至初始状态向量

        :return:
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return np.array(x, dtype=np.float32)
