import tensorflow as tf
import numpy as np
from TOOLS.Logger import Logger
import abc

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8


"""
为了实现策略网络模型，估值网络模型的具体定义形式与算法本体的相对独立，所以在模型文件中定义了
策略网络模型和估值网络模型的基本接口，个人实现的网络模型需要满足这一接口规范
"""


class AbsControlModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_model(self, ph_obs: tf.placeholder, act_max: np.ndarray, is_ou_noise: bool,
                    ph_ou_noise: tf.placeholder) -> tuple:
        """
        构建策略网络模型的计算图（基于状态观测向量占位符，返回行动向量计算节点）

        :param ph_obs: tf.placeholder,
            可以注入状态观测数据组的占位符
        :param act_max: np.ndarray,
            行动向量的最高限值向量
        :param is_ou_noise: bool,
            是否在随机采样过程中引入ou噪音
        :param ph_ou_noise: tf.placeholder,
            填写OU噪音的占位符
        :return: tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable],
            基于状态观测的行动向量决策分布的均值（mu），标准差（std），具体行动向量样本（pi），行动向量样本在决策分布中的对数似然
        """
        pass


class AbsEvaluateModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build_model(self, ph_obs: tf.placeholder, ph_act: tf.placeholder,
                    policy: tf.Variable) -> (tf.Variable, tf.Variable):
        """
        构建估值网络模型的计算图（基于状态观测向量占位符和行动向量占位符（或者行动向量计算节点），返回状态-行动估值计算节点）

        :param ph_obs: tf.placeholder,
            可以注入状态观测数据组的占位符
        :param ph_act: tf.placeholder,
            可以注入行动向量的占位符
        :param policy: tf.Variable,
            行动向量的计算节点
        :return: tuple[tf.Variable, tf.Variable],
            第一个返回值是状态-行动（占位符）估值计算节点，第二个返回值是状态-行动（计算节点）估值计算节点
        """
        pass


def mlp(x: tf.Variable, hidden_sizes: tuple = (32, ), activation: tf.function = tf.tanh,
        output_activation: tf.function = None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def gaussian_likelihood(x: tf.Variable, mu: tf.Variable, log_std: tf.Variable):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std) + EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu, pi, logp_pi):
    logp_pi -= tf.reduce_sum(2 * (np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


class MLPContiModel(AbsControlModel):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple, activation: str,
                 logger: Logger):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.logger = logger

        assert activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        # 确定激活函数
        if activation == 'Linear':
            self.activation = lambda x: x
        elif activation == 'Tanh':
            self.activation = tf.tanh
        elif activation == 'Sigmoid':
            self.activation = tf.nn.sigmoid
        elif activation == "ReLU":
            self.activation = tf.nn.relu

    def build_model(self, ph_obs: tf.placeholder, act_lim: np.ndarray,
                    is_ou_noise: bool, ph_ou_noise: tf.placeholder):
        hidden_layers = mlp(ph_obs, self.hidden_sizes, self.activation, self.activation)
        mu = tf.layers.dense(hidden_layers, units=self.act_dim, activation=None)
        log_std = tf.layers.dense(hidden_layers, units=self.act_dim, activation=None)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        if is_ou_noise:
            pi = mu + tf.tanh(ph_ou_noise) * std
        else:
            pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        mu *= act_lim
        pi *= act_lim
        return mu, std, pi, logp_pi


class MLPEvaluateModel(AbsEvaluateModel):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: tuple, hd_activation: str, logger: Logger):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_size
        self.logger = logger

        assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        # 确定激活函数
        if hd_activation == 'Linear':
            self.activation = lambda x: x
        elif hd_activation == 'Tanh':
            self.activation = tf.tanh
        elif hd_activation == 'Sigmoid':
            self.activation = tf.nn.sigmoid
        elif hd_activation == "ReLU":
            self.activation = tf.nn.relu

    def build_model(self, ph_obs: tf.placeholder, ph_act: tf.placeholder, policy: tf.Variable):
        for k, units in enumerate(self.hidden_sizes):
            if k == 0:
                obs_t = tf.concat([ph_obs, ph_act], axis=1)
                pol_obs_t = tf.concat([ph_obs, policy], axis=1)
                weight_shape = (self.obs_dim+self.act_dim, units)
            else:
                weight_shape = (self.hidden_sizes[k-1], units)
            bias_shape = (units, )
            weight = tf.get_variable(name='Weight_%d' % (k+1), shape=weight_shape, dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5))
            bias = tf.get_variable(name='Bias_%d' % (k+1), shape=bias_shape, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.3))
            obs_t = self.activation(tf.matmul(obs_t, weight) + bias)
            pol_obs_t = self.activation(tf.matmul(pol_obs_t, weight) + bias)
        output_weight_shape = (self.hidden_sizes[-1], 1)
        output_bias_shape = (1, )
        output_weight = tf.get_variable(name='Output_Weight', shape=output_weight_shape, dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(0.0, 0.6))
        output_bias = tf.get_variable(name='Output_bias', shape=output_bias_shape, dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.3))
        obs_t = tf.matmul(obs_t, output_weight) + output_bias
        pol_obs_t = tf.matmul(pol_obs_t, output_weight) + output_bias
        return tf.squeeze(obs_t, axis=[1]), tf.squeeze(pol_obs_t, axis=[1])


