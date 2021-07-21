import tensorflow as tf
from .model import AbsControlModel, AbsEvaluateModel
from ENVS.AbstractEnv import AbsEnv
from TOOLS.Logger import Logger
import numpy as np
import os


class PPOTDLambda(object):
    def __init__(self, env: AbsEnv, policy_model: AbsControlModel, evaluate_model: AbsEvaluateModel, model_dir: str,
                 exp_name: str, logger: Logger, gamma: float, clip_ratio: float, policy_lr: float, evaluate_lr: float,
                 conti_act_lim: np.ndarray):
        self.env = env
        self.policy_model = policy_model
        self.evaluate_model = evaluate_model
        self.target_dir_path = os.path.join(model_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        self.target_file_path = os.path.join(self.target_dir_path, 'PPOTDLambda_Model.ckpt')
        self.logger = logger
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_lr = policy_lr
        self.evaluate_lr = evaluate_lr
        self.conti_act_lim = conti_act_lim

        self.ph_obs_t, self.ph_act_t, self.ph_adv_t, self.ph_partial_real_rep, self.act_t, self.val_t, \
            self.update_params, self.adv_loss, self.train_pol_model, self.eva_loss, \
            self.train_eva_model = self.build_model()

    def build_model(self):
        # ++++++++ 占位符 ++++++++
        ph_obs_t = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='PH_Obs_t')
        ph_act_t = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='PH_Act_t')
        ph_adv_t = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='PH_Adv_t')
        ph_partial_real_rep = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='PH_Par_Real_Rep')
        # ++++++++ 占位符 ++++++++
        # ++++++++ 模型组 ++++++++
        main_policy_model_name = 'MainPolicyModel'
        pre_policy_model_name = 'PrePolicyModel'
        with tf.variable_scope(main_policy_model_name):
            main_pi = self.policy_model.build_model(ph_obs_t, trainable=True)
        with tf.variable_scope(pre_policy_model_name):
            pre_pi = self.policy_model.build_model(ph_obs_t, trainable=False)
        with tf.variable_scope('EvaluateModel'):
            val_t = self.evaluate_model.build_model(ph_obs_t)
        params_main_model = [i for i in tf.global_variables() if main_policy_model_name in i.name]
        params_pre_model = [i for i in tf.global_variables() if pre_policy_model_name in i.name]
        params_eva_model = [i for i in tf.global_variables() if 'EvaluateModel' in i.name]
        # ++++++++ 模型组 ++++++++
        # ++++++++ 算法组 ++++++++
        act_t = tf.clip_by_value(tf.squeeze(main_pi.sample(1), axis=0), -self.conti_act_lim, self.conti_act_lim)
        pro_ph_act = main_pi.prob(ph_act_t)
        pre_pro_ph_act = tf.stop_gradient(pre_pi.prob(ph_act_t))
        # 更新前置策略网络参数
        update_params = [param_pre.assign(param_main)
                         for param_main, param_pre in zip(params_main_model, params_pre_model)]
        # 重要性采样系数
        ratio = pro_ph_act / pre_pro_ph_act
        # 广义优势增进
        adv_up = ratio * ph_adv_t
        # 总体广义优势损失
        adv_loss = -tf.reduce_mean(tf.minimum(adv_up,
                                              tf.clip_by_value(ratio,
                                                               1.0-self.clip_ratio, 1.0+self.clip_ratio) * ph_adv_t))
        # 训练策略网络模型
        train_pol_model = tf.train.AdamOptimizer(learning_rate=self.policy_lr).minimize(adv_loss)
        eva_loss = tf.reduce_mean(tf.square(ph_partial_real_rep - val_t))
        train_eva_model = tf.train.AdamOptimizer(learning_rate=self.evaluate_lr).minimize(eva_loss)
        # ++++++++ 算法组 ++++++++

        return [ph_obs_t, ph_act_t, ph_adv_t, ph_partial_real_rep,
                act_t, val_t, update_params, adv_loss, train_pol_model, eva_loss, train_eva_model]

    def train(self, retrain_label: bool, max_iter_per_epoch: int, learning_epoch: int, mini_batch: int, update_num: int,
              save_freq: int):
        sess = tf.Session()
        model_saver = tf.train.Saver(max_to_keep=5)
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())

        def sample_episode():
            batch_obs = []
            batch_actions = []
            batch_rs = []
            for i in range(1):
                observation = self.env.reset()
                j = 0
                k = 0
                exp_repay = 0
                minibatch_obs = []
                minibatch_actions = []
                minibatch_rs = []
                while j < max_iter_per_epoch:
                    state = np.reshape(observation, [1, 3])
                    action = sess.run(self.act_t, feed_dict={self.ph_obs_t: state})[0]
                    reward, observation_, done, info = self.env.step(action)
                    exp_repay += reward
                    # 存储当前观测
                    minibatch_obs.append(np.reshape(observation, [1, 3])[0, :])
                    # 存储当前动作
                    minibatch_actions.append(action)
                    # 存储立即回报
                    minibatch_rs.append((reward+8)/8)
                    k = k + 1
                    j = j + 1
                    if k == mini_batch or j == max_iter_per_epoch:
                        # 处理回报
                        reward_sum = sess.run(self.val_t, feed_dict={self.ph_obs_t: observation_.reshape(1, -1)})[0, 0]
                        discouted_sum_reward = np.zeros_like(minibatch_rs)
                        for t in reversed(range(0, len(minibatch_rs))):
                            reward_sum = reward_sum * self.gamma + minibatch_rs[t]
                            discouted_sum_reward[t] = reward_sum
                        # 将mini批的数据存储到批回报中
                        for t in range(len(minibatch_rs)):
                            batch_rs.append(discouted_sum_reward[t])
                            batch_obs.append(minibatch_obs[t])
                            batch_actions.append(minibatch_actions[t])
                        k = 0
                        minibatch_obs = []
                        minibatch_actions = []
                        minibatch_rs = []
                    # 智能体往前推进一步
                    observation = observation_
            # reshape 观测和回报
            batch_obs = np.reshape(batch_obs, [len(batch_obs), self.env.obs_dim])
            batch_actions = np.reshape(batch_actions, [len(batch_actions), self.env.act_dim])
            batch_rs = np.reshape(batch_rs, [len(batch_rs), 1])
            return batch_obs, batch_actions, batch_rs

        for epoch in range(learning_epoch):
            batch_obs_t, batch_act_t, batch_rew_t = sample_episode()
            # 将前置网络的参数设置为当前主网络状态
            sess.run(self.update_params)
            # 计算出广义优势
            adv = batch_rew_t - sess.run(self.val_t, feed_dict={self.ph_obs_t: batch_obs_t})
            for _ in range(update_num):
                sess.run([self.adv_loss, self.train_pol_model], feed_dict={self.ph_obs_t: batch_obs_t,
                                                                           self.ph_act_t: batch_act_t,
                                                                           self.ph_adv_t: adv})
            for _ in range(update_num):
                sess.run([self.eva_loss, self.train_eva_model], feed_dict={self.ph_obs_t: batch_obs_t,
                                                                           self.ph_partial_real_rep: batch_rew_t})
            cur_exp_repay = self.test(1, max_iter_per_epoch, sess)
            if epoch == 0:
                smoothed_exp_repay = cur_exp_repay
            else:
                smoothed_exp_repay = 0.95*smoothed_exp_repay + 0.05 * cur_exp_repay
            print('[%d] 该回合的平滑得分为：%.2f' % (epoch, smoothed_exp_repay))
            if epoch > 1 and epoch % save_freq == 0:
                self.logger.to_log('存储模型参数')
                model_saver.save(sess, self.target_file_path, global_step=epoch)
        self.logger.to_log('模型学习结束，存储模型参数')
        model_saver.save(sess, self.target_file_path, global_step=learning_epoch)

    def test(self, learning_epoch, max_iter_per_epoch, sess):
        #sess = tf.Session()
        #model_saver = tf.train.Saver(max_to_keep=5)
        for epoch in range(learning_epoch):
            obs_t = self.env.reset()
            exp_repay = 0.0
            for iter in range(max_iter_per_epoch):
                act_t = sess.run(self.act_t, feed_dict={self.ph_obs_t: obs_t.reshape((1, -1))})[0]
                rew_t, obs_n, don_n, _ = self.env.step(act_t)
                exp_repay += rew_t
                obs_t = obs_n

                if don_n:
                    #print('[测试]游戏正常结束，总得分为：%.2f' % exp_repay)
                    break
            else:
                #print('[测试]游戏达到最大控制次数，总得分为：%.2f' % exp_repay)
                pass
        return exp_repay
