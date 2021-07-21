import tensorflow as tf
import abc
from TOOLS.Logger import Logger


class AbsControlModel(metaclass=abc.ABCMeta):
    pass


class AbsEvaluateModel(metaclass=abc.ABCMeta):
    pass


class MLPContiControlModel(AbsControlModel):
    def __init__(self, obs_dim, act_dim, hidden_size, hd_activation, max_control_lim, logger):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation
        self.max_control_lim = max_control_lim
        self.logger = logger

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            print('[Error]: 目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
            print('         可选择的激活函数类型包括：' + ','.join(hd_activation))
            raise e

    def build_model(self, obs_ph: tf.placeholder, trainable: bool):
        # 确定激活函数
        if self.hd_activation == 'Linear':
            activation = lambda x: x
        elif self.hd_activation == 'Tanh':
            activation = tf.tanh
        elif self.hd_activation == 'Sigmoid':
            activation = tf.nn.sigmoid
        elif self.hd_activation == "ReLU":
            activation = tf.nn.relu

        for k, units in enumerate(self.hidden_size):
            if k == 0:
                x = obs_ph
            x = tf.layers.dense(inputs=x, units=units, activation=activation, trainable=trainable)
        mu = tf.layers.dense(inputs=x, units=self.act_dim, activation=tf.tanh, trainable=trainable)
        mu = mu * self.max_control_lim
        std = tf.layers.dense(inputs=x, units=self.act_dim, activation=tf.nn.softplus,
                              trainable=trainable)
        pi = tf.contrib.distributions.Normal(mu, std)
        return pi


class MLPEvaluateModel(AbsEvaluateModel):
    def __init__(self, obs_dim: int, hidden_size: tuple, hd_activation: str, logger: Logger):
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation
        self.logger = logger

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            print('[Error]: 目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
            print('         可选择的激活函数类型包括：' + ','.join(hd_activation))
            raise e

    def build_model(self, obs_ph):
        # 确定激活函数
        activation = None
        if self.hd_activation == 'Linear':
            activation = None
        elif self.hd_activation == 'Tanh':
            activation = tf.tanh
        elif self.hd_activation == 'Sigmoid':
            activation = tf.nn.sigmoid
        elif self.hd_activation == "ReLU":
            activation = tf.nn.relu

        x = obs_ph
        for k, units in enumerate(self.hidden_size):
            x = tf.layers.dense(x, units, activation=activation)
        x = tf.layers.dense(inputs=x, units=1)
        return x
