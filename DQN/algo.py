import numpy as np
import tensorflow as tf
from ENVS.AbstractEnv import AbsImageEnv
from .model import AbsDeepQNetwork
from TOOLS.Logger import Logger
import os


class ExperienceBuffer(object):
    def __init__(self, obs_dim: int, act_dim: int, buffer_size: int):
        self.obs_t_buffer = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.act_t_buffer = np.zeros(shape=(buffer_size, act_dim), dtype=np.float32)
        self.rew_t_buffer = np.zeros(shape=(buffer_size, ), dtype=np.float32)
        self.obs_n_buffer = np.zeros(shape=(buffer_size, *obs_dim), dtype=np.float32)
        self.don_n_buffer = np.zeros(shape=(buffer_size, ), dtype=np.float32)

        self.current_index = 0
        self.buffer_size = buffer_size

    def store(self, obs_t: np.ndarray, act_t: np.ndarray, rew_t: np.ndarray, obs_n: np.ndarray, don_n: np.ndarray):
        cur_index = self.current_index % self.buffer_size
        self.obs_t_buffer[cur_index] = obs_t
        self.act_t_buffer[cur_index] = act_t
        self.rew_t_buffer[cur_index] = rew_t
        self.obs_n_buffer[cur_index] = obs_n
        self.don_n_buffer[cur_index] = don_n

        self.current_index += 1

    def sample(self, sample_size):
        cur_index = min(self.current_index, self.buffer_size)
        sample_size = min(sample_size, cur_index)
        sample_index = np.random.choice(range(cur_index), size=sample_size, replace=False)
        return [
            self.obs_t_buffer[sample_index],
            self.act_t_buffer[sample_index],
            self.rew_t_buffer[sample_index],
            self.obs_n_buffer[sample_index],
            self.don_n_buffer[sample_index]
        ]


class DQNLearning(object):
    def __init__(self, env: AbsImageEnv, evaluate_model: AbsDeepQNetwork, logger: Logger, save_dir: str, exp_name: str,
                 rho: float, gamma: float, evaluate_lr: float):
        """
        用于解决基于图像进行强化学习训练的控制问题的深度Q网络学习算法

        :param env: AbsImageEnv,
            基于图像的环境对象
        :param evaluate_model: AbsDeepQNetwork,
            基于深度（卷积）网络的策略模型
        :param logger: Logger,
            日志对象
        :param save_dir: str,
            模型参数的存储文件夹
        :param exp_name: str,
            试验的名称
        :param rho: float,
            目标网络模型参数向主网络参数靠近的速度
        :param gamma: float,
            远期奖励折现比例
        """
        self.env = env
        self.evaluate_model = evaluate_model
        self.rho = rho
        self.gamma = gamma
        self.logger = logger
        self.evaluate_lr = evaluate_lr
        self.target_dir_path = os.path.join(save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        self.target_file_path = os.path.join(self.target_dir_path, 'DQN_Model.ckpt')

        self.obs_t_ph, self.obs_n_ph, self.act_t_ph, self.rew_t_ph, self.don_n_ph, self.init_evaluate_params, \
            self.update_evaluate_params, self.main_value, self.targ_value, self.train_evaluate = self.build_model()

    def build_model(self):
        # +++++++++++ 占位符 +++++++++++
        obs_t_ph = tf.placeholder(tf.float32, shape=(None, *self.env.obs_dim), name='Obs_t_PH')
        obs_n_ph = tf.placeholder(tf.float32, shape=(None, *self.env.obs_dim), name='Obs_n_PH')
        act_t_ph = tf.placeholder(tf.float32, shape=(None, self.env.act_dim), name='Act_t_PH')
        rew_t_ph = tf.placeholder(tf.float32, shape=(None, ),                 name='Rew_t_PH')
        don_n_ph = tf.placeholder(tf.float32, shape=(None, ),                 name='Don_n_PH')
        # +++++++++++ 占位符 +++++++++++
        # +++++++++++ 模型 +++++++++++++
        main_evaluate_name = 'MainEvaluateModel'
        target_evaluate_name = 'TargetEvaluateModel'
        with tf.variable_scope(main_evaluate_name):
            main_value = self.evaluate_model.build_model(obs_t_ph)
        with tf.variable_scope(target_evaluate_name):
            targ_value = self.evaluate_model.build_model(obs_n_ph)
        main_evaluate_params = [i for i in tf.global_variables() if main_evaluate_name in i.name]
        targ_evaluate_params = [i for i in tf.global_variables() if target_evaluate_name in i.name]
        # 目标网络模型参数向主网络模型参数靠近
        update_evaluate_params = [tf.assign(targ_param, self.rho*main_param + (1-self.rho)*targ_param)
            for main_param, targ_param in zip(main_evaluate_params, targ_evaluate_params)]
        # 初始化目标网络模型参数到主网络模型参数
        init_evaluate_params = [tf.assign(targ_param, main_param)
                                for main_param, targ_param in zip(main_evaluate_params, targ_evaluate_params)]

        # +++++++++++ 模型 +++++++++++++
        # +++++++++++ 算法 +++++++++++++
        partial_real_repay = rew_t_ph + self.gamma * (1-don_n_ph) * tf.reduce_max(targ_value, axis=1)
        partial_real_repay = tf.stop_gradient(partial_real_repay)
        item_main_value = tf.reduce_sum(tf.multiply(main_value, act_t_ph), axis=1)
        main_evaluate_loss = tf.losses.mean_squared_error(labels=partial_real_repay, predictions=item_main_value)
        train_evaluate = tf.train.AdamOptimizer(learning_rate=self.evaluate_lr).minimize(main_evaluate_loss,
                                                                             var_list=main_evaluate_params)
        return obs_t_ph, obs_n_ph, act_t_ph, rew_t_ph, don_n_ph, init_evaluate_params, update_evaluate_params,\
               main_value, targ_value, train_evaluate

    def train(self, learn_epochs: int, max_iter_per_epoch: int, is_retrain_label: bool, buffer_size: int,
              init_epsilon: float, final_epsilon: float, eps_de_coef: float = 0.995, learn_pre_num: int = 500,
              learn_freq: int = 5, sample_size: int = 200):

        def get_epsilon_greedy_action(obs_t, cur_epsilon):
            act_t = np.zeros([self.env.act_dim])
            if np.random.rand() < 1 - cur_epsilon:
                value_vec = sess.run(self.main_value, feed_dict={self.obs_t_ph: [obs_t]})[0]
                arg_max = np.argmax(value_vec)
                act_t[arg_max] = 1
            else:
                if np.random.rand() < 0.7:
                    arg_rand = 0
                else:
                    arg_rand = 1
                act_t[arg_rand] = 1
            return act_t

        # 创建计算资源对象
        sess = tf.Session()
        # 创建模型存储对象
        model_saver = tf.train.Saver(max_to_keep=5)
        if is_retrain_label:
            print(self.target_dir_path)
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_evaluate_params)
        # 创建数据缓存器
        data_buffer = ExperienceBuffer(self.env.obs_dim, self.env.act_dim, buffer_size)
        cur_epsilon = init_epsilon
        total_learn_count = 0
        ep_repay_list = []
        for epoch in range(learn_epochs):
            obs_t = self.env.reset()
            ep_repay = 0
            self.logger.to_log('当前的随机行动概率：%.4f' % cur_epsilon)
            for epoch_control_index in range(max_iter_per_epoch):
                act_t = get_epsilon_greedy_action(obs_t, cur_epsilon)
                if (epoch * max_iter_per_epoch + epoch_control_index) > learn_pre_num and cur_epsilon > final_epsilon:
                    cur_epsilon *= eps_de_coef
                rew_t, obs_n, don_n, _ = self.env.step(act_t)
                data_buffer.store(obs_t, act_t, rew_t, obs_n, don_n)
                ep_repay += rew_t

                if (epoch * max_iter_per_epoch + epoch_control_index) > learn_pre_num and \
                        epoch_control_index % learn_freq == 0:
                    obs_t_b, act_t_b, rew_t_b, obs_n_b, don_n_b = data_buffer.sample(sample_size)
                    sess.run(self.train_evaluate, feed_dict={self.obs_t_ph: obs_t_b, self.act_t_ph: act_t_b,
                                                             self.rew_t_ph: rew_t_b, self.obs_n_ph: obs_n_b,
                                                             self.don_n_ph: don_n_b})
                    total_learn_count += 1
                    if total_learn_count % 3 == 0:
                        sess.run(self.update_evaluate_params)

                    if total_learn_count % 6 == 0:
                        model_saver.save(sess, self.target_file_path, global_step=epoch)

                obs_t = obs_n
                if don_n:
                    self.logger.to_log('[%d]游戏正常结束，总得分为：%.2f' % (epoch, ep_repay))
                    break
            else:
                self.logger.to_log('[%d]游戏到达最大控制次数，总得分为：%.2f' % (epoch, ep_repay))

            ep_repay_list.append(ep_repay)
            if len(ep_repay_list) > 1000:
                self.logger.to_log('[%d]最近100次控制的平均得分为：%.2f' % (epoch, np.mean(ep_repay_list[-100:])))

    def test(self, test_epochs, max_iter_per_epoch):

        def get_greedy_action(obs_t):
            act_t = np.zeros([self.env.act_dim])
            value_vec = sess.run(self.main_value, feed_dict={self.obs_t_ph: [obs_t]})[0]
            arg_max = np.argmax(value_vec)
            act_t[arg_max] = 1
            return act_t

        sess = tf.Session()
        model_saver = tf.train.Saver(max_to_keep=5)
        model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        ep_repay_list = []
        for epoch in range(test_epochs):
            obs_t = self.env.reset()
            ep_repay = 0
            self.logger.to_log('[%d]开始进行控制' % epoch)
            for control_index in range(max_iter_per_epoch):
                try:
                    self.env.render()
                except Exception as e:
                    pass
                act_t = get_greedy_action(obs_t)
                rew_t, obs_n, don_n, _ = self.env.step(act_t)
                ep_repay += rew_t
                obs_t = obs_n

                if don_n:
                    self.logger.to_log('游戏正常结束，总得分为：%.2f' % ep_repay)
                    break
            else:
                self.logger.to_log('游戏达到最大控制次数，总得分为：%.2f' % ep_repay)
            ep_repay_list.append(ep_repay)
        self.logger.to_log('测试结束。平均得分为： %.2f' % np.mean(ep_repay_list))
