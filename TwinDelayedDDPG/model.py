import tensorflow as tf
from TOOLS.Logger import Logger
import numpy as np
import abc

"""
为了实现策略网络模型，估值网络模型的具体定义形式与算法本体的相对独立，所以在模型文件中定义了
策略网络模型和估值网络模型的基本接口，个人实现的网络模型需要满足这一接口规范
"""


class AbsControlModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_model(self, ph_obs: tf.placeholder, act_max: np.ndarray) -> tf.Variable:
        """
        构建策略网络模型的计算图（基于状态观测向量占位符，返回行动向量计算节点）

        :param ph_obs: tf.placeholder,
            可以注入状态观测数据组的占位符
        :param act_max: np.ndarray,
            行动向量的最高限值向量
        :return: tf.Variable,
            行动向量计算节点
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


class MLPContiModel(AbsControlModel):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple, hd_activation: str, logger: Logger):
        """
        基于多层感知机的连续控制策略网络

        :param obs_dim: int,
            状态观测向量的长度
        :param act_dim: int,
            行动向量的长度
        :param hidden_sizes: tuple,
            隐藏层的层数和每一层的神经元个数
        :param hd_activation: str,
            隐藏层使用的激活函数
        :param logger: Logger,
            记录相关信息的日志对象
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_sizes
        self.hd_activation = hd_activation
        self.logger = logger

        try:
            assert hd_activation in ['Linear', 'Tanh', 'Sigmoid', 'ReLU']
        except AssertionError as e:
            self.logger.to_warn('目前激活函数的类型尚不支持输入的类型：%s' % hd_activation)
            raise e

    def build_model(self, ph_obs: tf.placeholder, act_max: np.ndarray) -> tf.Variable:
        """
        构建策略网络的计算图

        :param ph_obs: tf.placeholder,
            用于传递状态观测向量的占位符
        :param act_max: np.ndarray,
            连续控制指令的上限数组
        :return: tf.Variable,
            基于状态观测向量，策略网络计算出来的控制指令
        """
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
                x = ph_obs
                weight_shape = (self.obs_dim, units)
                bias_shape = (units, )
            else:
                weight_shape = (self.hidden_size[k-1], units)
                bias_shape = (units, )
            # 定义隐藏层的参数
            hidden_weight = tf.get_variable(name='P_Weight_%d' % (k+1), shape=weight_shape, dtype=tf.float32,
                                            initializer=tf.random_normal_initializer(mean=0, stddev=0.8))
            hidden_bias = tf.get_variable(name='P_Bias_%d' % (k+1), shape=bias_shape, dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))
            x = activation(tf.matmul(x, hidden_weight) + hidden_bias)
        # 定义输出层映射
        output_weight = tf.get_variable(name='P_Weight_Output', shape=(self.hidden_size[-1], self.act_dim),
                                        dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.8))
        output_bias = tf.get_variable(name='P_Bias_Output', shape=(self.act_dim, ), dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0))
        mu = tf.add(tf.matmul(x, output_weight), output_bias)
        mu = tf.tanh(mu, name='Mean') * act_max
        return mu


class MLPEvaluateModel(AbsEvaluateModel):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: tuple, hd_activation: str, logger: Logger):
        """
        基于多层感知器的状态-动作估值网络

        :param obs_dim: int,
            状态观测向量的维度
        :param act_dim: int,
            行动向量的维度
        :param hidden_size: tuple,
            隐藏层的层数和每一层中神经单元的个数
        :param hd_activation: str,
            隐藏层的激活函数
        :param logger: Logger,
            用于记录相关信息的日志对象
        """
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

    def build_model(self, ph_obs: tf.placeholder, ph_act: tf.placeholder, policy: tf.Variable):
        """
        构建估值网络的计算图

        :param ph_obs: tf.placeholder,
            用于填充状态观测向量的占位符
        :param ph_act: tf.placeholder,
            用于填充行动向量的占位符
        :param policy: tf.Variable,
            代表策略网络给出的行动向量的计算节点
        :return: tuple,
            (基于给定状态观测向量和给定行动向量的估值计算节点, 基于给定状态观测向量和策略网络的估值计算节点)
        """
        # 确定激活函数
        activation = None
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
                x_act = tf.concat([ph_obs, ph_act], axis=1)
                x_policy = tf.concat([ph_obs, policy], axis=1)
                weight_shape = (self.obs_dim + self.act_dim, units)
            else:
                weight_shape = (self.hidden_size[k-1], units)

            bias_shape = (units, )
            weight = tf.get_variable(name='E_Weight_%d' % (k+1), shape=weight_shape, dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0, stddev=0.4))
            bias = tf.get_variable(name='E_Bias_%d' % (k+1), shape=bias_shape, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.3))
            x_act = activation(tf.matmul(x_act, weight) + bias)
            x_policy = activation(tf.matmul(x_policy, weight) + bias)
        output_weight_shape = (self.hidden_size[-1], 1)
        output_bias_shape = (1, )
        output_weight = tf.get_variable(name='E_Output_Weight', shape=output_weight_shape, dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(0.0, 0.4))
        output_bias = tf.get_variable(name='E_Output_Bias', shape=output_bias_shape, dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.4))
        x_act_out = tf.matmul(x_act, output_weight) + output_bias
        x_policy_out = tf.matmul(x_policy, output_weight) + output_bias
        return tf.squeeze(x_act_out, axis=[1]), tf.squeeze(x_policy_out, axis=[1])

