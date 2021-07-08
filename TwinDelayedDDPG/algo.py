import tensorflow as tf
import numpy as np
from ENVS.AbstractEnv import AbsEnv
from TOOLS.Logger import Logger
from TOOLS.noises import OrnsteinUhlenbeckActionNoise
from .model import AbsControlModel, AbsEvaluateModel
import random
import os
from tensorflow.train import AdamOptimizer


class DataBuffer(object):
    def __init__(self, obs_dim, act_dim, size):
        self.obs_t_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_t_buffer = np.zeros((size, act_dim), dtype=np.float32)
        self.reward_t_buffer = np.zeros((size,), dtype=np.float32)
        self.obs_n_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        self.done_buffer = np.zeros((size, ), dtype=np.float32)
        self.current_index = 0
        self.size = size

    def store(self, obs_t, act_t, reward_t, obs_n, done):
        self.current_index_ = self.current_index % self.size
        self.obs_t_buffer[self.current_index_] = obs_t
        self.act_t_buffer[self.current_index_] = act_t
        self.reward_t_buffer[self.current_index_] = reward_t
        self.obs_n_buffer[self.current_index_] = obs_n
        self.done_buffer[self.current_index_] = done
        self.current_index += 1

    def get(self, sample_size):
        sample_index = random.sample(range(min(self.current_index, self.size)), sample_size)

        return [self.obs_t_buffer[sample_index], self.act_t_buffer[sample_index], self.reward_t_buffer[sample_index],
                self.obs_n_buffer[sample_index], self.done_buffer[sample_index]]

    def get_buffer_size(self):
        return min(self.current_index, self.size)


class TwinDelayedDDPG(object):
    def __init__(self, env: AbsEnv, policy_model: AbsControlModel, evaluate_model: AbsEvaluateModel, save_dir: str,
                 exp_name: str, logger: Logger, gamma: float, eva_lr: float, pol_lr: float, rho: float,
                 conti_act_low: np.ndarray, conti_act_high: np.ndarray, is_OU_noise: bool):
        """
        双延时的DDPG算法可以有效解决常规DDPG算法存在的初期高估各个状态-动作价值的问题

        :param env: AbsEnv,
            与DDPG决策器进行交互的环境对象，这里使用的环境对象需要满足AbsEnv的接口规范
        :param policy_model: AbsControlModel,
            策略网络模型
        :param evaluate_model: AbsEvaluateModel,
            估值网络模型
        :param save_dir: str,
            存储模型参数的文件夹地址
        :param exp_name: str,
            试验名称
        :param logger: Logger,
            日志对象
        :param gamma: float,
            远期奖励的折现系数
        :param eva_lr: float,
            估值网络的学习速率
        :param pol_lr: float,
            策略网络的学习速率
        :param rho: float,
            目标网络向主网络靠近的速度
        :param conti_act_low: np.ndarray,
            行动向量的下限
        :param conti_act_high: np.ndarray,
            行动向量的上限
        :param is_OU_noise: bool,
            是否使用OU噪音
        """
        # 算法运行依赖的动态环境对象
        self.env = env
        # 策略网络模型
        self.policy_model = policy_model
        # 估值网络模型
        self.evaluate_model = evaluate_model
        # 远期奖励折算系数
        self.gamma = gamma
        # 估值网络的学习速率
        self.evaluate_lr = eva_lr
        # 策略网络的学习速率
        self.policy_lr = pol_lr
        # 目标网络向主网络靠近的速率
        self.rho = rho
        # 日志文件对象
        self.logger = logger
        # 模型参数存储文件夹
        self.target_dir_path = os.path.join(save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        # 模型参数存储文件名称
        self.target_file_name = os.path.join(self.target_dir_path, 'TwinDelayedDDPG_Model.ckpt')
        # 连续动作的上界和下界
        self.conti_act_high = conti_act_high
        self.conti_act_low = conti_act_low
        # 是否使用OU噪音
        self.is_OU_noise = is_OU_noise

        self.ph_obs, self.ph_act, self.ph_reward, self.ph_obs_new, self.ph_done, self.ph_act_to_obs_new, \
            self.init_target_params, self.update_target_params, self.policy, self.target_policy, \
            self.evaluate_1_loss, self.evaluate_2_loss, self.train_evaluate_op, self.policy_loss, \
            self.train_policy_op = self.build_model()

    def build_model(self):
        # ++++++  占位符  ++++++
        ph_reward = tf.placeholder(dtype=tf.float32, shape=(None, ), name='Reward_PH')
        ph_done = tf.placeholder(dtype=tf.float32, shape=(None, ), name='Done_PH')
        ph_obs_new = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='Obs_New_PH')
        ph_obs = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='Obs_PH')
        ph_act = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='Act_PH')
        ph_act_to_obs_new = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='Act_to_Obs_New_PH')
        # ++++++  占位符  ++++++
        #  +++++++++++++++++ 模型定义 ++++++++++++++++
        # 主策略网络的名称
        main_policy_name = 'MainPolicy'
        # 目标策略网络的名称
        target_policy_name = 'TargetPolicy'
        # 主估值网络的名称
        main_evaluate_1_name = 'MainEvaluate-1'
        main_evaluate_2_name = 'MainEvaluate-2'
        # 目标估值网络的名称
        target_evaluate_1_name = 'TargetEvaluate-1'
        target_evaluate_2_name = 'TargetEvaluate-2'
        with tf.variable_scope(main_policy_name):
            policy = self.policy_model.build_model(ph_obs, self.conti_act_high)
        with tf.variable_scope(target_policy_name):
            target_policy = self.policy_model.build_model(ph_obs_new, self.conti_act_high)
        with tf.variable_scope(main_evaluate_1_name):
            evaluate_1_act, evaluate_1_policy = self.evaluate_model.build_model(ph_obs, ph_act, policy)
        with tf.variable_scope(main_evaluate_2_name):
            evaluate_2_act, evaluate_2_policy = self.evaluate_model.build_model(ph_obs, ph_act, policy)
        with tf.variable_scope(target_evaluate_1_name):
            target_evaluate_1_act, target_evaluate_1_policy = self.evaluate_model.build_model(ph_obs_new,
                                                                                              ph_act_to_obs_new,
                                                                                              target_policy)
        with tf.variable_scope(target_evaluate_2_name):
            target_evaluate_2_act, target_evaluate_2_policy = self.evaluate_model.build_model(ph_obs_new,
                                                                                              ph_act_to_obs_new,
                                                                                              target_policy)
        # 主策略网络的全体参数
        policy_params = [i for i in tf.global_variables() if main_policy_name in i.name]
        # 主估值网络的全体参数
        evaluate_1_params = [i for i in tf.global_variables() if main_evaluate_1_name in i.name]
        evaluate_2_params = [i for i in tf.global_variables() if main_evaluate_2_name in i.name]
        # 目标策略网络的全体参数
        target_policy_params = [i for i in tf.global_variables() if target_policy_name in i.name]
        # 目标估值网络的全体参数
        target_evaluate_1_params = [i for i in tf.global_variables() if target_evaluate_1_name in i.name]
        target_evaluate_2_params = [i for i in tf.global_variables() if target_evaluate_2_name in i.name]
        #  +++++++++++++++++ 模型定义 ++++++++++++++++
        #  ++++++++++++++++++ 算法 ++++++++++++++++++
        minimum_evaluate = tf.minimum(target_evaluate_1_act, target_evaluate_2_act)
        partial_real_repay = ph_reward + self.gamma * (1 - ph_done) * minimum_evaluate
        partial_real_repay = tf.stop_gradient(partial_real_repay)
        evaluate_1_loss = tf.reduce_mean((evaluate_1_act - partial_real_repay) ** 2)
        evaluate_2_loss = tf.reduce_mean((evaluate_2_act - partial_real_repay) ** 2)
        train_evaluate_1_op = AdamOptimizer(learning_rate=self.evaluate_lr).minimize(evaluate_1_loss,
                                                                                     var_list=evaluate_1_params)
        train_evaluate_2_op = AdamOptimizer(learning_rate=self.evaluate_lr).minimize(evaluate_2_loss,
                                                                                     var_list=evaluate_2_params)
        train_evaluate_op = tf.group([train_evaluate_1_op, train_evaluate_2_op])
        policy_loss = -tf.reduce_mean(evaluate_1_policy)
        train_policy_op = AdamOptimizer(learning_rate=self.policy_lr).minimize(policy_loss, var_list=policy_params)
        # 更新目标网络参数
        update_target_params = []
        for main_params, target_params in [(policy_params, target_policy_params),
                                           (evaluate_1_params, target_evaluate_1_params),
                                           (evaluate_2_params, target_evaluate_2_params)]:
            update_target_params.append(
                tf.group([tf.assign(target_param, self.rho * main_param + (1-self.rho) * target_param)
                          for main_param, target_param in zip(main_params, target_params)])
            )
        # 初始化目标网络参数
        init_target_params = []
        for main_params, target_params in [(policy_params, target_policy_params),
                                           (evaluate_1_params, target_evaluate_1_params),
                                           (evaluate_2_params, target_evaluate_2_params)]:
            init_target_params.append(
                tf.group([tf.assign(target_param, main_param)
                          for main_param, target_param in zip(main_params, target_params)])
            )
        #  ++++++++++++++++++ 算法 ++++++++++++++++++
        return ph_obs, ph_act, ph_reward, ph_obs_new, ph_done, ph_act_to_obs_new, init_target_params, \
            update_target_params, policy, target_policy, evaluate_1_loss, evaluate_2_loss, train_evaluate_op, \
            policy_loss, train_policy_op

    def train(self, buffer_size: int = 1000000, retrain_label: bool = False, learn_epochs: int = 150,
              max_iter_per_epoch: int = 3000, sample_size: int = 250, save_freq: int = 5, noise_scale: float = 0.1,
              start_steps: int = 10000, update_after: int = 10000, update_every: int = 200):
        """
        使用Twin Delayed DDPG算法对策略网络和估值网络进行训练

        :param buffer_size: int,
            数据缓存器的容量
        :param retrain_label: bool,
            是否使用存储的模型参数来重置模型
        :param learn_epochs: int,
            本次训练进行的回合总数
        :param max_iter_per_epoch: int,
            在每一次控制回合中最多下达的控制决策次数
        :param sample_size: int,
            每一次学习中从数据缓存器中抽取的样本数量
        :param save_freq: int,
            每间隔save_freq次学习后就保存一次网络模型参数
        :param noise_scale: float,
            在训练过程中每一个控制决策所伴随的随机扰动的标准差
        :param start_steps: int,
            在进行start_steps次随机控制之后才开始进行由决策网络下达控制指令
        :param update_after: int,
            在进行update_after次控制之后才开始进行模型参数学习
        :param update_every: int,
            每间隔update_every次控制之后进行一次模型参数学习
        :return:
        """
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型存储对象
        model_saver = tf.train.Saver(max_to_keep=5)
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target_params)
        # 创建数据缓存对象
        data_buffer = DataBuffer(self.env.obs_dim, self.env.act_dim, buffer_size)
        # 开始进行模拟控制和模型学习

        def get_action(obs_t: np.ndarray, noise_scale: float):
            act_t_ = sess.run(self.policy, feed_dict={self.ph_obs: obs_t.reshape(1, -1)})[0]
            if not self.is_OU_noise:
                act_t = act_t_ + noise_scale * np.random.randn(self.env.act_dim)
            else:
                act_t = act_t_ + OU_noise()
            return np.clip(act_t, self.conti_act_low, self.conti_act_high)

        def get_action_n(obs_n: np.ndarray, noise_scale: float):
            act_n_ = sess.run(self.target_policy, feed_dict={self.ph_obs_new: obs_n})
            act_n = act_n_ + noise_scale * np.random.randn(*act_n_.shape)
            return np.clip(act_n, self.conti_act_low, self.conti_act_high)

        total_control_count = 0
        for epoch in range(learn_epochs):
            obs_t, ep_repay = self.env.reset(), 0
            if self.is_OU_noise:
                OU_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.env.act_dim, ), sigma=0.75)
            for epoch_control_index in range(max_iter_per_epoch):
                if total_control_count < start_steps:
                    if not self.is_OU_noise:
                        act_t = self.conti_act_high * (np.random.rand(self.env.act_dim) * 2 - 1)
                    else:
                        act_t = OU_noise()
                else:
                    act_t = get_action(obs_t, noise_scale)

                reward_t, obs_n, done_n, _ = self.env.step(act_t)
                ep_repay += reward_t
                total_control_count += 1

                data_buffer.store(obs_t, act_t, reward_t, obs_n, done_n)
                obs_t = obs_n

                if done_n:
                    self.logger.to_log('[%d]本轮游戏正常结束，总得分为：%.2f' % (epoch, ep_repay))
                    break
                elif epoch_control_index == max_iter_per_epoch-1:
                    self.logger.to_log('[%d]本轮游戏已到达最大控制次数，总得分为：%.2f' % (epoch, ep_repay))

                # 进行模型学习
                if total_control_count >= update_after and total_control_count % update_every == 0:
                    for j in range(update_every):
                        obs_t_buf, act_t_buf, rew_t_buf, obs_n_buf, done_n_buf = data_buffer.get(sample_size)
                        act_n_buf = get_action_n(obs_n_buf, noise_scale/4.0)
                        feed_dict = {self.ph_obs: obs_t_buf, self.ph_act: act_t_buf, self.ph_reward: rew_t_buf,
                                     self.ph_obs_new: obs_n_buf, self.ph_done: done_n_buf,
                                     self.ph_act_to_obs_new: act_n_buf}
                        eva_outs = sess.run([self.evaluate_1_loss, self.evaluate_2_loss, self.train_evaluate_op],
                                            feed_dict=feed_dict)
                        if j % 3 == 0:
                            pol_outs = sess.run([self.policy_loss, self.train_policy_op],
                                                feed_dict={self.ph_obs: obs_t_buf})
                            sess.run(self.update_target_params)

                    if total_control_count % (40*update_every) == 0:
                        self.logger.to_log('  估值网络-1的损失为：%.2f' % eva_outs[0])
                        self.logger.to_log('  估值网络-2的损失为：%.2f' % eva_outs[1])
                        self.logger.to_log('  策略网络的损失为： %.2f' % pol_outs[0])

                    if np.random.rand() < 1/save_freq:
                        self.logger.to_log('  存储网络模型')
                        model_saver.save(sess, self.target_file_name, global_step=epoch)

    def test(self, test_epochs: int, max_iter_per_epoch: int) -> None:
        """
        对Twin Delayed DDPG训练好的策略网络模型进行测试

        :param test_epochs: int,
            总的测试回合数量
        :param max_iter_per_epoch: int,
            每一个回合中最大控制次数
        :return: None,
        """
        import time

        sess = tf.Session()
        model_saver = tf.train.Saver()
        model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        for epoch in range(test_epochs):
            self.logger.to_log('[%d]开始进行测试' % epoch)
            obs_t = self.env.reset()
            ep_repay = 0
            for k in range(max_iter_per_epoch):
                try:
                    self.env.render()
                    time.sleep(0.03)
                except:
                    pass
                act_t = sess.run(self.policy, feed_dict={self.ph_obs: obs_t.reshape((1, -1))})[0]
                reward_t, obs_n, done_n, _ = self.env.step(act_t)
                ep_repay += reward_t
                obs_t = obs_n
                if done_n or k == max_iter_per_epoch-1:
                    self.logger.to_log('控制回合结束，总得分为：%.2f' % ep_repay)
                    break

