import tensorflow as tf
import numpy as np
import os
from TOOLS.Logger import Logger
from ENVS.AbstractEnv import AbsEnv
from .model import AbsEvaluateModel, AbsControlModel


class DataBuffer(object):
    def __init__(self, obs_dim: int, act_dim: int, buffer_size: int, gamma: float):
        self.obs_t_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.act_t_buffer = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rew_t_buffer = np.zeros((buffer_size, ), dtype=np.float32)
        self.obs_n_buffer = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.don_n_buffer = np.zeros((buffer_size, ), dtype=np.float32)
        self.rep_t_buffer = np.zeros((buffer_size, ), dtype=np.float32)

        self.gamma = gamma

        self.current_index = 0
        self.path_start_index = 0
        self.buffer_size = buffer_size

    def store(self, obs_t, act_t, rew_t, obs_n, don_n):
        cur_index = self.current_index
        self.obs_t_buffer[cur_index] = obs_t
        self.act_t_buffer[cur_index] = act_t
        self.rew_t_buffer[cur_index] = rew_t
        self.obs_n_buffer[cur_index] = obs_n
        self.don_n_buffer[cur_index] = don_n

        self.current_index += 1

    def finish_path(self, last_value):
        stage_index = slice(self.path_start_index, self.current_index)
        stage_rew = self.rew_t_buffer[stage_index]
        discount_rep = []
        cur_discount_rep = last_value[0]
        for rew_t in reversed(stage_rew):
            cur_discount_rep = self.gamma * cur_discount_rep + rew_t
            discount_rep.append(cur_discount_rep)
        self.rep_t_buffer[stage_index] = np.array(discount_rep[::-1])

        self.path_start_index = self.current_index

    def get_data(self):
        assert self.current_index == self.buffer_size
        self.current_index, self.path_start_index = 0, 0
        return self.obs_t_buffer, self.act_t_buffer, self.rep_t_buffer, self.don_n_buffer


class PPOTDLamda(object):
    def __init__(self, env: AbsEnv, policy_model: AbsControlModel, evaluate_model: AbsEvaluateModel,
                 model_save_dir: str, exp_name: str, gamma: float, logger: Logger, clip_ratio: float,
                 policy_lr: float, evaluate_lr: float, is_ou_noise: bool, act_lim: np.ndarray):
        self.env = env
        self.policy_model = policy_model
        self.evaluate_model = evaluate_model
        self.target_dir_path = os.path.join(model_save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        self.target_file_path = os.path.join(self.target_dir_path, 'PPOTDLamda_Model.ckpt')
        self.gamma = gamma
        self.logger = logger
        self.clip_ratio = clip_ratio
        self.policy_lr = policy_lr
        self.evaluate_lr = evaluate_lr
        self.is_ou_noise = is_ou_noise
        self.act_lim = act_lim

        self.obs_t_ph, self.act_ph, self.adv_ph, self.partial_real_rep_ph, self.val, self.act_t, \
            self.update_pre_policy_params, self.adv_loss, self.eva_loss, self.train_adv_op, \
            self.train_eva_op = self.build_model()

    def build_model(self):
        # +++++++++++++++ 占位符 ++++++++++++++++++
        obs_t_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='Obs_PH')
        act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='Act_PH')
        adv_ph = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='Adv_PH')
        partial_real_rep_ph = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='Partial_Real_Repay_PH')
        # +++++++++++++++ 占位符 ++++++++++++++++++
        # +++++++++++++++ 模  型 ++++++++++++++++++
        val = self.evaluate_model.build_model(obs_t_ph)
        with tf.variable_scope('MainPolicyModel'):
            main_pi = self.policy_model.build_model(obs_t_ph, trainable=True)
        with tf.variable_scope('PrePolicyModel'):
            pre_pi = self.policy_model.build_model(obs_t_ph, trainable=False)
        main_policy_params = [i for i in tf.global_variables() if 'MainPolicyModel' in i.name]
        pre_policy_params = [i for i in tf.global_variables() if 'PrePolicyModel' in i.name]
        update_pre_policy_params = [tf.assign(pre_param, main_param)
                                    for main_param, pre_param in zip(main_policy_params, pre_policy_params)]
        act_t = tf.clip_by_value(tf.squeeze(main_pi.sample(1), axis=0), -self.act_lim, self.act_lim)
        pro_ph_act = main_pi.prob(act_ph)
        pre_pro_ph_act = tf.stop_gradient(pre_pi.prob(act_ph))
        # +++++++++++++++ 模  型 ++++++++++++++++++
        # +++++++++++++++ 算  法 ++++++++++++++++++
        ratio = pro_ph_act / pre_pro_ph_act
        adv_adj = ratio * adv_ph
        adv_loss = -tf.reduce_mean(tf.minimum(adv_adj, tf.clip_by_value(ratio,
                                                                        1-self.clip_ratio, 1+self.clip_ratio) * adv_ph))
        train_adv_op = tf.train.AdamOptimizer(learning_rate=self.policy_lr).minimize(adv_loss,
                                                                                     var_list=main_policy_params)
        eva_loss = tf.reduce_mean(tf.square(partial_real_rep_ph - val))
        train_eva_op = tf.train.AdamOptimizer(learning_rate=self.evaluate_lr).minimize(eva_loss)
        # +++++++++++++++ 算  法 ++++++++++++++++++
        return obs_t_ph, act_ph, adv_ph, partial_real_rep_ph, val, act_t, update_pre_policy_params, adv_loss, eva_loss,\
            train_adv_op, train_eva_op

    def train(self, learning_epoch, max_iter_per_epoch, mini_batch, retrain_label, update_num, save_freq):
        sess = tf.Session()
        model_saver = tf.train.Saver()
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())
            sess.run(self.update_pre_policy_params)
        data_buffer = DataBuffer(self.env.obs_dim, self.env.act_dim, max_iter_per_epoch, self.gamma)
        for epoch in range(learning_epoch):
            obs_t = self.env.reset()
            ep_repay = 0
            for epoch_control_index in range(max_iter_per_epoch):
                act_t = sess.run(self.act_t, feed_dict={self.obs_t_ph: obs_t.reshape(1, -1)})[0]
                rew_t, obs_n, don_n, _ = self.env.step(act_t)
                rew_t = (rew_t + 8) / 8
                ep_repay += rew_t
                data_buffer.store(obs_t, act_t, rew_t, obs_n, don_n)
                obs_t = obs_n

                if (don_n or epoch_control_index % mini_batch == 0 or epoch_control_index == max_iter_per_epoch-1)\
                        and epoch_control_index > 0:
                    if don_n:
                        last_value = 0
                    else:
                        last_value = sess.run(self.val, feed_dict={self.obs_t_ph: obs_n.reshape(1, -1)})[0]
                    data_buffer.finish_path(last_value)

                    if don_n:
                        self.logger.to_log('游戏正常结束，总得分为：%.2f' % ep_repay)
                    elif epoch_control_index == max_iter_per_epoch - 1:
                        self.logger.to_log('游戏达到最大控制次数，总得分为：%.2f' % ep_repay)

                    if don_n:
                        obs_t = self.env.reset()
                        ep_repay = 0
            obs_t_buffer, act_t_buffer, rep_t_buffer, don_n_buffer = data_buffer.get_data()
            rep_t_buffer = np.reshape(rep_t_buffer, (len(rep_t_buffer), 1))
            don_n_buffer = np.reshape(don_n_buffer, (len(don_n_buffer), 1))
            val_t_buffer = sess.run(self.val, feed_dict={self.obs_t_ph: obs_t_buffer})
            adv_t_buffer = rep_t_buffer - val_t_buffer
            '''
            mean = np.mean(adv_t_buffer)
            std = np.std(adv_t_buffer)
            adv_t_buffer = (adv_t_buffer-mean) / std
            '''
            adv_loss_list = []
            eva_loss_list = []
            for _ in range(update_num):
                _, adv_loss = sess.run([self.train_adv_op, self.adv_loss],
                                       feed_dict={self.obs_t_ph: obs_t_buffer, self.act_ph: act_t_buffer,
                                                  self.adv_ph: adv_t_buffer})
                _, eva_loss = sess.run([self.train_eva_op, self.eva_loss],
                                       feed_dict={self.obs_t_ph: obs_t_buffer,
                                                  self.partial_real_rep_ph: rep_t_buffer})
                adv_loss_list.append(adv_loss)
                eva_loss_list.append(eva_loss)
            self.logger.to_log('[%d]广义优势损失为：%.4f    估值损失为：%.4f' % (epoch, np.mean(adv_loss_list),
                                                                                np.mean(eva_loss_list)))
            sess.run(self.update_pre_policy_params)
            if epoch > 1 and epoch % save_freq == 0:
                model_saver.save(sess, self.target_file_path, global_step=epoch)
        self.logger.to_log('学习过程结束，存储最终的模型参数')
        model_saver.save(sess, self.target_file_path, global_step=learning_epoch)


