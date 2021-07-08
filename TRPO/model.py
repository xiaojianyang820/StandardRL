import tensorflow as tf
import numpy as np


class MLPCateModel(object):
    def __init__(self, obs_dim, act_dim, hidden_size, hd_activation, logger):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation
        self.logger = logger

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            self.logger.to_warn('目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
            raise e

    def kl_divergence(self, logp_0, logp_1):
        all_kls = tf.reduce_sum(tf.exp(logp_1) * (logp_1 - logp_0), axis=1)
        return tf.reduce_mean(all_kls)

    def build_model(self, obs_ph, act_ph, old_info_dict):
        # 确定激活函数
        activation = None
        if self.hd_activation == 'Linear':
            activation = lambda x: x
        elif self.hd_activation == 'Tanh':
            activation = tf.tanh
        elif self.hd_activation == 'Sigmoid':
            activation = tf.nn.sigmoid
        elif self.hd_activation == 'ReLU':
            activation = tf.nn.relu

        # 定义隐藏层映射
        for k, units in enumerate(self.hidden_size):
            if k == 0:
                x = obs_ph
                weight_shape = (self.obs_dim, units)
                bias_shape = (units,)
            else:
                weight_shape = (self.hidden_size[k - 1], units)
                bias_shape = (units,)
            # 定义隐藏层的参数
            hidden_weight = tf.get_variable(name='P_weight_%d' % (k + 1), shape=weight_shape, dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
            hidden_bias = tf.get_variable(name='P_bias_%d' % (k + 1), shape=bias_shape, dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.5))
            x = activation(tf.matmul(x, hidden_weight) + hidden_bias)
        # 定义输出层映射
        output_weight = tf.get_variable(name='P_weight_output', shape=(self.hidden_size[-1], self.act_dim),
                                        dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.8))
        output_bias = tf.get_variable(name='P_bias_output', shape=(self.act_dim,), dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0))
        output = tf.matmul(x, output_weight) + output_bias
        # 行动向量计算节点
        pi = tf.squeeze(tf.random.categorical(output, num_samples=1), axis=1)
        pi = tf.one_hot(pi, depth=self.act_dim, axis=1, name='Action')
        # 决策分布
        logp_vec = tf.nn.log_softmax(output)
        # 前置策略网络下的决策分布
        logp_vec_old_ph = old_info_dict['logp_vec']
        kl_divergence = self.kl_divergence(logp_vec, logp_vec_old_ph)
        # 行动向量的对数似然计算节点
        logp_pi = tf.reduce_sum(tf.multiply(pi, logp_vec), axis=1, name='Action_Logp')
        # 基于给定行动向量占位符的对数似然计算节点
        logp_act_ph = tf.reduce_sum(tf.multiply(act_ph, logp_vec), axis=1, name='Act_PH_Logp')
        # 决策分布信息
        info_dict = {'logp_vec': logp_vec}

        return pi, logp_pi, logp_act_ph, info_dict, kl_divergence


class MLPContiModel(object):
    EPS = 1e-8
    def __init__(self, obs_dim, act_dim, hidden_size, hd_activation, act_lim, logger):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation
        self.act_lim = act_lim
        self.logger = logger

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            self.logger.to_warn('目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
            raise e

    def diagonal_gaussian_kl(self, mu_0, log_std_0, mu_1, log_std_1):
        var_0, var_1 = tf.exp(2 * log_std_0), tf.exp(2 * log_std_1)
        pre_sum = 0.5 * (((mu_1 - mu_0) ** 2 + var_0)/(var_1 + self.EPS) - 1) + log_std_1 - log_std_0
        all_kls = tf.reduce_mean(pre_sum, axis=1)
        return tf.reduce_mean(all_kls)

    def gaussian_likelihood(self, act, mu, log_std):
        pre_sum = -0.5 * (((act - mu) / (tf.exp(log_std) + self.EPS)) ** 2 + 2*log_std + tf.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def build_model(self, obs_ph, act_ph, old_info_dict):
        # 确定激活函数
        activation = None
        if self.hd_activation == 'Linear':
            activation = lambda x: x
        elif self.hd_activation == 'Tanh':
            activation = tf.tanh
        elif self.hd_activation == 'Sigmoid':
            activation = tf.nn.sigmoid
        elif self.hd_activation == 'ReLU':
            activation = tf.nn.relu

        for k, units in enumerate(self.hidden_size):
            if k == 0:
                x = obs_ph
            x = tf.layers.dense(x, units=units, activation=activation)
        mu = tf.layers.dense(x, units=self.act_dim, activation=None, name='Mean')
        mu = tf.clip_by_value(mu, -self.act_lim, self.act_lim)
        log_std = tf.get_variable(name='Log_STD', shape=(self.act_dim, ), dtype=tf.float32,
                                  initializer=tf.constant_initializer(-0.5))
        std = tf.exp(log_std)
        info_dict = {
            'Mean': mu, 'Log_STD': log_std
        }
        old_mu = old_info_dict['Mean']
        old_log_std = old_info_dict['Log_STD']
        kl_divergence = self.diagonal_gaussian_kl(mu, log_std, old_mu, old_log_std)
        pi = tf.add(mu, tf.random_normal(tf.shape(mu)) * std, name='Action')
        logp_act_ph = self.gaussian_likelihood(act_ph, mu, log_std)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)
        return pi, logp_pi, logp_act_ph, info_dict, kl_divergence


class MLPEvaluateModel(object):
    def __init__(self, obs_dim, hidden_size, hd_activation, logger):
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.hd_activation = hd_activation
        self.logger = logger

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            self.logger.to_warn('目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
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
            if k == len(self.hidden_size):
                activation_ = None
            else:
                activation_ = activation
            x = tf.layers.dense(x, units=units, activation=activation_)
        x = tf.squeeze(x, axis=[1])
        return x
