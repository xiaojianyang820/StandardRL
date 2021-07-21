import abc
import tensorflow as tf


class AbsDeepQNetwork(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build_model(self, obs_t_ph: tf.placeholder):
        pass


class DeepQNetwork(AbsDeepQNetwork):
    def __init__(self, act_dim: int,
                 conv_1_filter_num: int = 32, conv_1_kernel_size: tuple = (8, 8), conv_1_strides: int = 4,
                 conv_2_filter_num: int = 64, conv_2_kernel_size: tuple = (4, 4), conv_2_strides: int = 2,
                 conv_3_filter_num: int = 64, conv_3_kernel_size: tuple = (3, 3), conv_3_strides: int = 1,
                 pool_size: tuple = (2, 2), pool_strides: int = 2, hd_activation=tf.nn.relu, fc_units=512):
        self.act_dim = act_dim
        self.conv_1_filter_num = conv_1_filter_num
        self.conv_1_kernel_size = conv_1_kernel_size
        self.conv_1_strides = conv_1_strides
        self.conv_2_filter_num = conv_2_filter_num
        self.conv_2_kernel_size = conv_2_kernel_size
        self.conv_2_strides = conv_2_strides
        self.conv_3_filter_num = conv_3_filter_num
        self.conv_3_kernel_size = conv_3_kernel_size
        self.conv_3_strides = conv_3_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.hd_activation = hd_activation
        self.fc_units = fc_units

    def build_model(self, obs_t_ph: tf.placeholder):
        hd_conv_1 = tf.layers.conv2d(inputs=obs_t_ph, filters=self.conv_1_filter_num, kernel_size=self.conv_1_kernel_size,
                                     strides=self.conv_1_strides, padding='same', activation=self.hd_activation,
                                     kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.01))
        hd_pool_1 = tf.layers.max_pooling2d(hd_conv_1, pool_size=self.pool_size, strides=self.pool_strides,
                                            padding='same')
        hd_conv_2 = tf.layers.conv2d(inputs=hd_pool_1, filters=self.conv_2_filter_num, kernel_size=self.conv_2_kernel_size,
                                     strides=self.conv_2_strides, padding='same', activation=self.hd_activation,
                                     kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.01))
        hd_conv_3 = tf.layers.conv2d(inputs=hd_conv_2, filters=self.conv_3_filter_num, kernel_size=self.conv_3_kernel_size,
                                     strides=self.conv_3_strides, padding='same', activation=self.hd_activation,
                                     kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.01))
        hd_conv_3_flat = tf.layers.flatten(hd_conv_3)
        hd_fc_1 = tf.layers.dense(hd_conv_3_flat, units=self.fc_units, activation=self.hd_activation,
                                  kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.01))
        value = tf.layers.dense(inputs=hd_fc_1, units=self.act_dim,
                                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                bias_initializer=tf.constant_initializer(0.01))
        return value
