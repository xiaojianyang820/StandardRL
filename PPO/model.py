import tensorflow as tf
import numpy as np
import abc
from TOOLS.Logger import Logger


class AbsMLPControlModel(metaclass=abc.ABCMeta):
    def build_model(self, ph_obs: tf.placeholder, ph_act: tf.placeholder, is_ou_noise: bool,
                    ou_noise_ph: tf.placeholder) -> tuple:
        pass


class AbsMLPEvaluateModel(metaclass=abc.ABCMeta):
    def build_model(self, ph_obs: tf.placeholder) -> tf.Variable:
        pass


class MLPCateModel(AbsMLPControlModel):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: tuple, hd_activation: str, logger: Logger,
                 conti_control_max: np.ndarray):
        """
        多层全连接神经网络的基本配置参数

        :param obs_dim: int,
            状态观测向量的维度
        :param act_dim: int,
            决策行动向量的维度
        :param hidden_size: tuple[int],
            隐藏层的形状
        :param hd_activation: str,
            隐藏层的激活函数类型
        :param logger: Logger,
            日志对象
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation
        self.logger = logger

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            print('[Error]: 目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
            print('         可选择的激活函数类型包括：' + ','.join(hd_activation))
            raise e

    def build_model(self, obs_ph: tf.placeholder, act_ph: tf.placeholder, is_ou_noise: bool,
                    ou_noise_ph: tf.placeholder) -> tuple:
        """
        基于外部所提供的状态观测占位符和行动向量占位符构建相应的计算节点张量
        :param obs_ph: tf.placeholder,
            状态观测的占位符，形状为[None, obs_dim]
        :param act_ph: tf.placeholder,
            行动向量的占位符，形状为[None, act_dim]
        :return: tuple,
            (基于状态观测计算出来的行动向量，该行动向量的对数似然，给定状态观测下特定行动向量的对数似然)
        """
        # 确定激活函数
        if self.hd_activation == 'Linear':
            activation = lambda x: x
        elif self.hd_activation == 'Tanh':
            activation = tf.tanh
        elif self.hd_activation == 'Sigmoid':
            activation = tf.nn.sigmoid
        elif self.hd_activation == "ReLU":
            activation = tf.nn.relu
        # 定义隐藏层映射
        for k, units in enumerate(self.hidden_size):
            if k == 0:
                x = obs_ph
                weight_shape = (self.obs_dim, units)
                bias_shape = (units, )
            else:
                weight_shape = (self.hidden_size[k-1], units)
                bias_shape = (units, )
            # 定义隐藏层的参数
            hidden_weight = tf.get_variable(name='P_weight_%d' % (k+1), shape=weight_shape, dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            hidden_bias = tf.get_variable(name='P_bias_%d' % (k+1), shape=bias_shape, dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.5))
            x = activation(tf.matmul(x, hidden_weight) + hidden_bias)
        # 定义输出层映射
        output_weight = tf.get_variable(name='P_weight_output', shape=(self.hidden_size[-1], self.act_dim),
                                        dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.8))
        output_bias = tf.get_variable(name='P_bias_output', shape=(self.act_dim, ), dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0))
        output = tf.matmul(x, output_weight) + output_bias
        # 行动向量计算节点
        pi = tf.squeeze(tf.random.categorical(output, num_samples=1), axis=[1])
        pi = tf.one_hot(pi, depth=self.act_dim, axis=1, name="Action")
        # 输出的对数似然向量计算节点
        logp_vec = tf.nn.log_softmax(output)
        # 行动向量对数似然值计算节点
        logp_pi = tf.reduce_sum(tf.multiply(pi, logp_vec, name="Action_LogP"), axis=1)
        # 基于特定行动向量占位符的对数似然值计算节点
        logp_act_ph = tf.reduce_sum(tf.multiply(act_ph, logp_vec, name="Act_PH_LogP"), axis=1)
        return pi, logp_pi, logp_act_ph


class MLPContiModel(AbsMLPControlModel):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: tuple, hd_activation: str, logger: Logger,
                 conti_control_max: np.ndarray):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation
        self.logger = logger
        self.conti_control_max = conti_control_max

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            print('[Error]: 目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
            print('         可选择的激活函数类型包括：' + ','.join(hd_activation))
            raise e

    def gaussian_likelihood(self, act: tf.Variable, mu: tf.Variable, log_std: tf.Variable) -> tf.Variable:
        pre_sum = -0.5 * (((act - mu)/(tf.exp(log_std) + 1e-8))**2 + 2*log_std + tf.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def build_model(self, obs_ph: tf.placeholder, act_ph: tf.placeholder, is_ou_noise: bool,
                    ou_noise_ph: tf.placeholder) -> tuple:
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
                weight_shape = (self.obs_dim, units)
                bias_shape = (units, )
            else:
                weight_shape = (self.hidden_size[k-1], units)
                bias_shape = (units, )
            # 定义隐藏层的参数
            hidden_weight = tf.get_variable(name='P_weight_%d' % (k+1), shape=weight_shape, dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            hidden_bias = tf.get_variable(name='P_bias_%d' % (k+1), shape=bias_shape, dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))
            x = activation(tf.matmul(x, hidden_weight) + hidden_bias)
        # 定义输出层映射
        output_weight = tf.get_variable(name='P_weight_output', shape=(self.hidden_size[-1], self.act_dim),
                                            dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.8))
        output_bias = tf.get_variable(name='P_bias_output', shape=(self.act_dim,), dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))
        mu = tf.add(tf.matmul(x, output_weight), output_bias, name='Mean')
        log_std = tf.get_variable(name='Log_STD', shape=(self.act_dim, ), dtype=tf.float32,
                                  initializer=tf.constant_initializer(-1))
        std = tf.exp(log_std)
        if is_ou_noise:
            pi = tf.add(mu, ou_noise_ph, name='Action')
        else:
            pi = tf.add(mu, tf.random_normal(tf.shape(mu)) * std, name='Action')
        pi = tf.clip_by_value(pi, -self.conti_control_max, self.conti_control_max)
        #pi = tf.tanh(pi) * self.conti_control_max
        logp_act_ph = self.gaussian_likelihood(act_ph, mu, log_std)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)
        return pi, logp_pi, logp_act_ph


class MLPEvaluateModel(AbsMLPEvaluateModel):
    def __init__(self, obs_dim: int, hidden_size: tuple, hd_activation: str, logger: Logger):
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation

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