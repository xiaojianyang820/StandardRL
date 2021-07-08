import numpy as np
import tensorflow as tf
import os
from tensorflow.train import AdamOptimizer
from TOOLS.Logger import Logger
from .model import AbsControlModel, AbsEvaluateModel
from ENVS.AbstractEnv import AbsEnv
from TOOLS.noises import OrnsteinUhlenbeckActionNoise


class ReplayBuffer(object):
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs_t_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs_n_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_t_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_t_buf = np.zeros([size, ], dtype=np.float32)
        self.done_n_buf = np.zeros([size, ], dtype=np.float32)
        self.cur_index, self.cur_size, self.max_size = 0, 0, size

    def store(self, obs_t: np.ndarray, act_t: np.ndarray, rew_t: np.ndarray, obs_n: np.ndarray, done_n: np.ndarray):
        cur_index = self.cur_size % self.max_size
        self.obs_t_buf[cur_index] = obs_t
        self.obs_n_buf[cur_index] = obs_n
        self.act_t_buf[cur_index] = act_t
        self.rew_t_buf[cur_index] = rew_t
        self.done_n_buf[cur_index] = done_n

        self.cur_size += 1

    def sample_batch(self, batch_size: int = 32):
        index_list = np.random.randint(0, min(self.cur_size, self.max_size), size=batch_size)
        return [self.obs_t_buf[index_list], self.act_t_buf[index_list], self.rew_t_buf[index_list],
                self.obs_n_buf[index_list], self.done_n_buf[index_list]]


class SAC(object):
    def __init__(self, env: AbsEnv, policy_model: AbsControlModel, evaluate_model: AbsEvaluateModel, logger: Logger,
                 model_save_dir: str, exp_name: str, gamma: float, rho: float, alpha: float, pol_lr: float,
                 eva_lr: float, conti_control_max: np.ndarray, is_OU_noise: bool):
        # 当前智能体所面临的环境对象
        self.env = env
        # 策略网络模型
        self.policy_model = policy_model
        # 估值网络模型
        self.evaluate_model = evaluate_model
        # 日志对象
        self.logger = logger
        # 网络模型参数的存储位置
        self.target_dir_path = os.path.join(model_save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        self.target_file_path = os.path.join(self.target_dir_path, 'SAC_Model.ckpt')
        # 远期奖励的折扣系数
        self.gamma = gamma
        # 目标网络向主网络靠近的速度
        self.rho = rho
        # 对决策信息熵的奖励调节系数
        self.alpha = alpha
        # 策略网络的学习速率
        self.pol_lr = pol_lr
        # 估值网络的学习速率
        self.eva_lr = eva_lr
        # 连续控制指令的上限
        self.conti_control_max = conti_control_max
        # 是否在随机操作上使用OU噪音
        self.is_OU_noise = is_OU_noise

        self.ph_obs_t, self.ph_act_t, self.ph_rew_t, self.ph_obs_n, self.ph_don_n, self.ph_ou_noise, self.mu, self.std, self.pi,\
            self.logp_pi, self.train_pol_op, self.train_eva_op, self.update_targ_params, self.init_targ_params, \
            self.train_ops, self.main_pol_loss, self.main_eva_loss = self.build_model()

    def build_model(self):
        obs_dim = self.env.obs_dim
        act_dim = self.env.act_dim
        # ++++++++++ 占位符 ++++++++++
        ph_obs_t = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name='PH_Obs_C')
        ph_act_t = tf.placeholder(dtype=tf.float32, shape=(None, act_dim), name='PH_Act_C')
        ph_rew_t = tf.placeholder(dtype=tf.float32, shape=(None,), name='PH_Rew_C')
        ph_obs_n = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name="PH_Obs_n")
        ph_don_n = tf.placeholder(dtype=tf.float32, shape=(None,), name='PH_Done')
        ph_ou_noise = tf.placeholder(dtype=tf.float32, shape=(None, act_dim), name='PH_OU_Noise')
        # ++++++++++ 占位符 ++++++++++
        # ++++++++++ 构建模型 ++++++++++
        policy_model_name = 'Policy'
        evaluate_model_1_name = 'MainEvaluate-1'
        evaluate_model_2_name = 'MainEvaluate-2'
        target_evaluate_model_1_name = 'TargetEvaluate-1'
        target_evaluate_model_2_name = 'TargetEvaluate-2'
        with tf.variable_scope(policy_model_name):
            mu, std, pi, logp_pi = self.policy_model.build_model(ph_obs_t, self.conti_control_max, self.is_OU_noise,
                                                                 ph_ou_noise)
        with tf.variable_scope(policy_model_name, reuse=True):
            mu_n, std_n, pi_n, logp_pi_n = self.policy_model.build_model(ph_obs_n, self.conti_control_max, False,
                                                                         ph_ou_noise)
        with tf.variable_scope(evaluate_model_1_name):
            main_eva_1_act, main_eva_1_pol = self.evaluate_model.build_model(ph_obs_t, ph_act_t, pi)
        with tf.variable_scope(evaluate_model_2_name):
            main_eva_2_act, main_eva_2_pol = self.evaluate_model.build_model(ph_obs_t, ph_act_t, pi)
        with tf.variable_scope(target_evaluate_model_1_name):
            _, targ_eva_1_pol = self.evaluate_model.build_model(ph_obs_n, ph_act_t, pi_n)
        with tf.variable_scope(target_evaluate_model_2_name):
            _, targ_eva_2_pol = self.evaluate_model.build_model(ph_obs_n, ph_act_t, pi_n)

        def get_vars(scope):
            return [x for x in tf.trainable_variables() if scope in x.name]

        # 主策略网络参数
        main_pol_params = get_vars(policy_model_name)
        # 主估值网络参数
        main_eva_1_params = get_vars(evaluate_model_1_name)
        main_eva_2_params = get_vars(evaluate_model_2_name)
        # 目标估值网络参数
        targ_eva_1_params = get_vars(target_evaluate_model_1_name)
        targ_eva_2_params = get_vars(target_evaluate_model_2_name)
        # ++++++++++ 构建模型 ++++++++++
        # ++++++++++ 算法 ++++++++++
        # 混合部分真实的远期目标估计
        min_targ_eva = tf.minimum(targ_eva_1_pol, targ_eva_2_pol)
        partial_real_repay = ph_rew_t + self.gamma * (1 - ph_don_n) * (min_targ_eva - self.alpha * logp_pi_n)
        partial_real_repay = tf.stop_gradient(partial_real_repay)
        # 策略网络损失以及训练操作
        min_main_eva = tf.minimum(main_eva_1_pol, main_eva_2_pol)
        main_pol_loss = -tf.reduce_mean(min_main_eva - self.alpha * logp_pi)
        train_pol_op = AdamOptimizer(learning_rate=self.pol_lr).minimize(main_pol_loss, var_list=main_pol_params)
        # 主估值网络损失以及训练操作
        main_eva_1_loss = tf.reduce_mean((main_eva_1_act - partial_real_repay) ** 2)
        main_eva_2_loss = tf.reduce_mean((main_eva_2_act - partial_real_repay) ** 2)
        train_eva_1_op = AdamOptimizer(learning_rate=self.eva_lr).minimize(main_eva_1_loss,
                                                                           var_list=main_eva_1_params)
        train_eva_2_op = AdamOptimizer(learning_rate=self.eva_lr).minimize(main_eva_2_loss,
                                                                           var_list=main_eva_2_params)
        train_eva_op = tf.group([train_eva_1_op, train_eva_2_op])
        # ++++++++++ 算法 ++++++++++
        # 目标估值网络参数向主估值网络参数靠近
        update_targ_1_params = [tf.assign(targ_eva_p, self.rho * main_eva_p + (1 - self.rho) * targ_eva_p)
                                for main_eva_p, targ_eva_p in zip(main_eva_1_params, targ_eva_1_params)]
        update_targ_2_params = [tf.assign(targ_eva_p, self.rho * main_eva_p + (1 - self.rho) * targ_eva_p)
                                for main_eva_p, targ_eva_p in zip(main_eva_2_params, targ_eva_2_params)]
        update_targ_params = tf.group([update_targ_1_params, update_targ_2_params])
        # 初始化目标估值网络参数
        init_targ_1_params = [tf.assign(targ_eva_p, main_eva_p)
                              for main_eva_p, targ_eva_p in zip(main_eva_1_params, targ_eva_1_params)]
        init_targ_2_params = [tf.assign(targ_eva_p, main_eva_p)
                              for main_eva_p, targ_eva_p in zip(main_eva_2_params, targ_eva_2_params)]
        init_targ_params = tf.group([init_targ_1_params, init_targ_2_params])

        step_ops = [main_pol_loss, main_eva_1_loss, main_eva_2_loss, train_pol_op, train_eva_op]

        return ph_obs_t, ph_act_t, ph_rew_t, ph_obs_n, ph_don_n, ph_ou_noise, mu, std, pi, logp_pi, train_pol_op, train_eva_op, \
            update_targ_params, init_targ_params, step_ops, main_pol_loss, main_eva_1_loss + main_eva_2_loss

    def get_action(self, obs_t: np.ndarray, deterministic: bool, sess: tf.Session,
                   ou_noise: OrnsteinUhlenbeckActionNoise):
        act_t_op = self.mu if deterministic else self.pi
        return sess.run(act_t_op, feed_dict={self.ph_obs_t: obs_t.reshape(1, -1),
                                             self.ph_ou_noise: ou_noise().reshape(1, -1)})[0]

    def train(self, learn_epochs: int, max_iter_per_epoch: int, retrain_label: bool, buffer_size: int,
              start_control_step: int, update_freq: int, sample_size: int, save_freq: int) -> None:
        """
        使用SAC算法对策略网络模型和估值网络模型进行训练

        :param learn_epochs: int,
            训练回合总数
        :param max_iter_per_epoch: int,
            每一个回合中最大控制次数
        :param retrain_label: bool,
            是否需要将网络模型参数重置为已有最新的训练参数
        :param buffer_size: int,
            数据缓存器大小
        :param start_control_step: int,
            多少次控制之后开始学习
        :param update_freq: int,
            策略网络和估值网络的参数更新频率
        :param sample_size: int,
            每一次学习过程中采样轨迹数量
        :param save_freq: int,
            策略网络和估值网络的参数保存频率
        :return: None,
        """
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型存储对象
        model_saver = tf.train.Saver(max_to_keep=5)
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_targ_params)
        # 创建数据缓存器
        data_buffer = ReplayBuffer(self.env.obs_dim, self.env.act_dim, buffer_size)

        # 开始控制测试和模型学习
        total_control_count = 0
        total_learn_count = 0
        for epoch in range(learn_epochs):
            if self.is_OU_noise:
                ou_noise = OrnsteinUhlenbeckActionNoise(np.zeros((self.env.act_dim, )), sigma=0.6)
            obs_t, ep_repay = self.env.reset(), 0
            for epoch_control_index in range(max_iter_per_epoch):
                # 产生行动向量决策
                if total_control_count < start_control_step:
                    if self.is_OU_noise:
                        act_t = np.tanh(ou_noise()) * self.conti_control_max
                    else:
                        act_t = np.random.randn(self.env.act_dim) * self.conti_control_max
                else:
                    act_t = self.get_action(obs_t, False, sess, ou_noise)
                act_t = np.clip(act_t, -self.conti_control_max, self.conti_control_max)
                # 执行行动决策
                rew_t, obs_n, don_n, _ = self.env.step(act_t)
                # 累加回合奖励
                ep_repay += rew_t
                # 记录历史数据
                data_buffer.store(obs_t, act_t, rew_t, obs_n, don_n)
                # 更新环境状态观测
                obs_t = obs_n
                total_control_count += 1

                if total_control_count % update_freq == 0 and total_control_count > start_control_step:
                    pol_loss_list = []
                    eva_loss_list = []
                    for j in range(update_freq):
                        batch_data = data_buffer.sample_batch(sample_size)
                        obs_t_b, act_t_b, rew_t_b, obs_n_b, don_n_b = batch_data
                        ou_noise_b = np.random.randn(*act_t_b.shape)
                        # 更新策略网络参数
                        pol_loss, _ = sess.run([self.main_pol_loss, self.train_pol_op],
                                               feed_dict={self.ph_obs_t: obs_t_b,
                                                          self.ph_ou_noise: ou_noise_b})
                        pol_loss_list.append(pol_loss)
                        # 更新估值网络参数
                        eva_loss, _ = sess.run([self.main_eva_loss, self.train_eva_op],
                                               feed_dict={self.ph_obs_t: obs_t_b,
                                                          self.ph_act_t: act_t_b,
                                                          self.ph_rew_t: rew_t_b,
                                                          self.ph_obs_n: obs_n_b,
                                                          self.ph_don_n: don_n_b})
                        eva_loss_list.append(eva_loss)
                        if j % 2 == 0:
                            sess.run(self.update_targ_params)

                        total_learn_count += 1
                    if total_learn_count % save_freq == 0:
                        self.logger.to_log('估值网络模型损失为：%.4f; 策略网络模型损失为：%.4f' % (np.mean(eva_loss_list),
                                                                               np.mean(pol_loss_list)))
                        self.logger.to_log('存储模型参数')
                        model_saver.save(sess, self.target_file_path, global_step=total_learn_count)

                if don_n:
                    self.logger.to_log('[%d]控制回合正常结束，总得分为： %.2f' % (epoch, ep_repay))
                    break
                if epoch_control_index == max_iter_per_epoch - 1:
                    self.logger.to_log('[%d]控制回合达到最大控制次数，总得分为：%.2f' % (epoch, ep_repay))

    def test(self, test_epochs: int, max_iter_per_epoch: int) -> None:
        """
        基于不同的游戏环境对使用SAC算法训练好的策略网络进行测试

        :param test_epochs: int,
            测试回合总数
        :param max_iter_per_epoch: int,
            每一回合中的最大控制次数
        :return: None,
        """
        import time
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型存储对象
        model_saver = tf.train.Saver()
        # 向计算资源会话中恢复模型参数
        model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))

        for epoch in range(test_epochs):
            self.logger.to_log('[%d]开始进行测试' % epoch)
            obs_t = self.env.reset()
            ep_repay = 0
            for k in range(max_iter_per_epoch):
                try:
                    self.env.render()
                    time.sleep(0.02)
                except Exception as e:
                    pass

                act_t = sess.run(self.mu, feed_dict={self.ph_obs_t: obs_t.reshape(1, -1)})
                rew_t, obs_n, don_n, _ = self.env.step(act_t[0])
                ep_repay += rew_t
                obs_t = obs_n
                if don_n or k == max_iter_per_epoch - 1:
                    self.logger.to_log('控制回合结束，总得分为：%.2f' % ep_repay)
                    break

