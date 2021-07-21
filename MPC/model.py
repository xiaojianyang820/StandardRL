import tensorflow as tf
import abc
from TOOLS.Logger import Logger


class AbsControlModel(metaclass=abc.ABCMeta):
    def build_model(self, ph_obs_t: tf.placeholder):
        pass


class AbsEvaluateModel(metaclass=abc.ABCMeta):
    def build_model(self, ph_obs_t: tf.placeholder):
        pass


class MLPContiControlModel(AbsControlModel):
    def __init__(self, obs_dim, act_dim, hidden_sizes, hd_activation, act_lim, logger):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.hd_activation = hd_activation
        self.act_lim = act_lim
        self.logger = logger

        assert hd_activation in ['ReLU', 'Sigmoid', 'Tanh']

    def build_model(self, ph_obs_t: tf.placeholder):
        activation = None
        if self.hd_activation == 'ReLU':
            activation = tf.nn.relu
        elif self.hd_activation == 'Sigmoid':
            activation = tf.nn.sigmoid
        elif self.hd_activation == 'Tanh':
            activation = tf.tanh

        x = ph_obs_t
        for units in self.hidden_sizes:
            x = tf.layers.dense(inputs=x, units=units, activation=activation)
        mu = tf.layers.dense(inputs=x, units=self.act_dim, activation=tf.nn.tanh)
        mu = mu * self.act_lim
        std = tf.layers.dense(inputs=x, units=self.act_dim, activation=tf.nn.softplus)
        normal_dist = tf.contrib.distributions.Normal(mu, std)
        act = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), -self.act_lim, self.act_lim)
        return normal_dist, act


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
        if self.hd_activation == 'Linear':
            activation = lambda x: x
        elif self.hd_activation == 'Tanh':
            activation = tf.tanh
        elif self.hd_activation == 'Sigmoid':
            activation = tf.nn.sigmoid
        elif self.hd_activation == "ReLU":
            activation = tf.nn.relu

        for k, units in enumerate(list(self.hidden_size) + [1]):
            if k == 0:
                x = obs_ph
                weight_shape = (self.obs_dim, units)
                bias_shape = (units, )
            else:
                weight_shape = (self.hidden_size[k-1], units)
                bias_shape = (units, )
            hidden_weight = tf.get_variable(name='V_weight_%d' % (k+1), shape=weight_shape, dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.3))
            hidden_bias = tf.get_variable(name='V_bias_%d' % (k+1), shape=bias_shape, dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.5))
            if k < len(self.hidden_size):
                x = activation(tf.matmul(x, hidden_weight) + hidden_bias)
            else:
                x = tf.matmul(x, hidden_weight) + hidden_bias

        x = tf.squeeze(x, axis=[1])
        return x
