import tensorflow as tf
import numpy as np
import os
from tensorflow.train import AdamOptimizer
from scipy import signal
from TOOLS.Logger import Logger
from .model import AbsMLPEvaluateModel, AbsMLPControlModel
from ENVS.AbstractEnv import AbsEnv
from TOOLS.noises import OrnsteinUhlenbeckActionNoise


class PPOBuffer(object):
    @classmethod
    def assist_func_1(cls, seq, coeff):
        return signal.lfilter([1], [1, float(-coeff)], seq[::-1], axis=0)[::-1]

    def __init__(self, obs_dim, act_dim, size, gamma, beta):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.gamma = gamma
        self.beta = beta

        self.current_index = 0
        self.con_epoch_start_index = 0

        # 状态数据序列
        self.obs_buffer = np.zeros(shape=(size, obs_dim), dtype=np.float32)
        # 行动数据序列
        self.act_buffer = np.zeros(shape=(size, act_dim), dtype=np.float32)
        # 行动的对数概率数据序列
        self.logp_act_buffer = np.zeros(shape=(size,), dtype=np.float32)
        # 奖励数据序列
        self.reward_buffer = np.zeros(shape=(size, ), dtype=np.float32)
        # 估值数据序列
        self.val_buffer = np.zeros(shape=(size, ), dtype=np.float32)
        # 回报数据序列
        self.repay_buffer = np.zeros(shape=(size, ), dtype=np.float32)
        # 广义优势数据序列
        self.adv_buffer = np.zeros(shape=(size, ), dtype=np.float32)

    def store(self, obs, act, rew, val, logp_act):
        self.obs_buffer[self.current_index] = obs
        self.act_buffer[self.current_index] = act
        self.logp_act_buffer[self.current_index] = logp_act
        self.reward_buffer[self.current_index] = rew
        self.val_buffer[self.current_index] = val

        self.current_index += 1

    def finish(self, last_value):
        stage_index = slice(self.con_epoch_start_index, self.current_index)
        reward_sq = self.reward_buffer[stage_index]
        val_sq = self.val_buffer[stage_index]
        # 将最后一个估值追加到奖励序列和估值序列里
        reward_sq = np.append(reward_sq, last_value)
        val_sq = np.append(val_sq, last_value)
        # 计算该阶段回报序列
        repay_sq = self.assist_func_1(reward_sq, self.gamma)[:-1]
        # 计算该阶段即刻优势序列
        deltas = reward_sq[:-1] + self.gamma * val_sq[1:] - val_sq[:-1]
        # 计算该阶段广义优势序列
        adv_sq = self.assist_func_1(deltas, self.gamma * self.beta)

        self.repay_buffer[stage_index] = repay_sq
        self.adv_buffer[stage_index] = adv_sq

        self.con_epoch_start_index = self.current_index

    def get(self):
        try:
            assert self.current_index == self.size
        except AssertionError as e:
            print("[Error]: 当前数据缓存器尚未饱和， 不宜进行训练")
            raise e
        self.current_index, self.con_epoch_start_index = 0, 0
        # 由于目前的估值网络不准确，可能导致广义优势整体上偏高或者偏低，所以需要从广义优势中移除均值和标准差
        adv_mean, adv_std = np.mean(self.adv_buffer), np.std(self.adv_buffer)
        self.adv_buffer = (self.adv_buffer - adv_mean)/adv_std
        return [self.obs_buffer, self.act_buffer, self.logp_act_buffer, self.repay_buffer, self.adv_buffer]


class PPO(object):
    def __init__(self, env: AbsEnv, policy_model: AbsMLPControlModel, evaluate_model: AbsMLPEvaluateModel,
                 model_save_dir: str, exp_name: str, gamma: float, beta: float, logger: Logger, clip_ratio: float,
                 policy_lr: float, evaluate_lr: float, is_ou_noise: bool):
        # 环境对象
        self.env = env
        # 策略网络
        self.policy_model = policy_model
        # 估值网络
        self.evaluate_model = evaluate_model
        # 日志对象
        self.logger = logger
        # 模型存储地址
        self.target_dir_path = os.path.join(model_save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        self.target_file_path = os.path.join(self.target_dir_path, 'PPO_Model.ckpt')
        # 远期奖励折现比率
        self.gamma = gamma
        # 远期的广义优势折现比率
        self.beta = beta
        # 样本广义优势增长比率截断数值
        self.clip_ratio = clip_ratio
        # 策略网络的学习速率
        self.policy_lr = policy_lr
        # 估值网络的学习速率
        self.evaluate_lr = evaluate_lr
        # 是否使用OU噪音
        self.is_ou_noise = is_ou_noise

        self.obs_ph, self.act_ph, self.logp_act_old_ph, self.adv_ph, self.repay_ph, self.ou_noise_ph, \
            self.val_op, self.pi_op, self.logp_pi_op, self.train_policy_op,\
            self.train_evaluate_op = self.build_model()

    def build_model(self):
        # ++++++++++++++++++++  占位符  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 定义状态观测的占位符和行动向量的占位符
        obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name="Obs_PH")
        act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name="Act_PH")
        # 定义前置策略所给出的行动对数似然占位符
        logp_act_old_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name="Logp_Act_Old_PH")
        # 定义广义优势的占位符
        adv_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='Adv_PH')
        # 定义实际回报的占位符
        repay_ph = tf.placeholder(dtype=tf.float32, shape=(None,), name='Repay_PH')
        # OU_Noise噪音样本占位符
        ou_noise_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='OU_Noise_PH')
        # ++++++++++++++++++++  占位符  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++  模型  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pi, logp_pi, logp_act_ph = self.policy_model.build_model(obs_ph, act_ph, self.is_ou_noise, ou_noise_ph)
        val = self.evaluate_model.build_model(obs_ph)
        # ++++++++++++++++++++  模型  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++  PPO算法  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 策略的差异比率系数
        ratio = tf.exp(logp_act_ph - logp_act_old_ph)
        # 广义策略优势
        adv_cum = ratio * adv_ph
        # 广义策略优势的上界
        # 之前在TRPO算法中，设计了复杂的计算逻辑来保证本次模型参数调整所带来的决策分布变化的KL散度的期望不超过阈值delta
        # 这一计算逻辑过于复杂，所以在PPO算法中简化这一逻辑。策略参数更新一定是向提高广义策略优势的方向进行的，所以当样本行动
        # 所对应的优势数值为正数的话，那么调整之后策略就会提高该状态下这一样本行动出现的概率，相应的ratio就会大于1。反之，会降低
        # 这一样本行动出现的概率，相应的ratio会小于1。这里就关心KL散度的具体变化，硬性的给定系数增长的上限和降低的下限。进而
        # 截断广义优势的增长。
        adv_max = tf.where(condition=adv_ph > 0, x=(1+self.clip_ratio) * adv_ph,
                           y=(1-self.clip_ratio) * adv_ph)
        # 广义策略优势损失
        policy_loss = -tf.reduce_mean(tf.minimum(adv_cum, adv_max))
        train_policy = AdamOptimizer(learning_rate=self.policy_lr).minimize(policy_loss)
        # 估值损失
        evaluate_loss = tf.reduce_mean(tf.square(val - repay_ph))
        train_evaluate = AdamOptimizer(learning_rate=self.evaluate_lr).minimize(evaluate_loss)
        # ++++++++++++++++++++  PPO算法  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return obs_ph, act_ph, logp_act_old_ph, adv_ph, repay_ph, ou_noise_ph, val, pi, logp_pi, \
               train_policy, train_evaluate

    def train(self, learning_epochs: int, retrain_label: bool, update_num: int,
              save_freq: int, max_iter_per_epoch: int):
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型保存对象
        model_saver = tf.train.Saver()
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())
        # 创建数据缓存器对象
        buffer = PPOBuffer(self.env.obs_dim, self.env.act_dim, max_iter_per_epoch, self.gamma, self.beta)
        for epoch in range(learning_epochs):
            ou_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.env.act_dim), 0.6)

            obs_t = self.env.reset()
            ep_repay = 0
            for k in range(max_iter_per_epoch):
                # 基于状态观测利用估值网络对当前环境状态进行估值
                val_t = sess.run(self.val_op, feed_dict={self.obs_ph: obs_t.reshape(1, -1)})
                # 基于状态观测进行决策，并产生样本行动向量
                act_t, logp_act_t = sess.run([self.pi_op, self.logp_pi_op],
                                             feed_dict={self.obs_ph: obs_t.reshape(1, -1),
                                                        self.ou_noise_ph: ou_noise().reshape(1, -1)})
                rew_t, obs_n, don_n, _ = self.env.step(act_t[0])
                ep_repay += rew_t
                # 向数据缓存器中写入本次控制的结果数据
                buffer.store(obs=obs_t, act=act_t, val=val_t, logp_act=logp_act_t, rew=rew_t)
                obs_t = obs_n

                if don_n or k == max_iter_per_epoch - 1:
                    if don_n:
                        self.logger.to_log('[%d]游戏正常结束，得分为：%.2f' % (epoch, ep_repay))
                    if k == max_iter_per_epoch - 1:
                        self.logger.to_log('[%d]游戏到达最大控制次数，得分为：%.2f' % (epoch, ep_repay))
                    last_val = 0 if don_n else sess.run(self.val_op, feed_dict={self.obs_ph: obs_n.reshape(1, -1)})[0]
                    buffer.finish(last_val)
                    obs_t = self.env.reset()
                    ep_repay = 0

                    if don_n and k != max_iter_per_epoch-1:
                        continue

                    # 读取缓存数据
                    obs_t_b, act_t_b, logp_act_t_b, repay_t_b, adv_t_b = buffer.get()
                    # 多次更新策略网络
                    for _ in range(update_num):
                        sess.run([self.train_policy_op, self.train_evaluate_op],
                                  feed_dict={self.obs_ph: obs_t_b, self.act_ph: act_t_b,
                                             self.logp_act_old_ph: logp_act_t_b, self.adv_ph: adv_t_b,
                                             self.repay_ph: repay_t_b})
            # 存储网络模型
            if (epoch+1) % save_freq == 0:
                self.logger.to_log('存储网络参数')
                model_saver.save(sess, self.target_file_path, global_step=epoch)
        self.logger.to_log('学习过程结束')
        self.logger.to_log('存储网络参数')
        model_saver.save(sess, self.target_file_path, global_step=learning_epochs)

    def test(self, test_epochs, max_iter_per_epoch):
        import time
        sess = tf.Session()
        model_saver = tf.train.Saver()
        model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        for epoch in range(test_epochs):
            self.logger.to_log('[%d]开始进行测试' % epoch)
            ep_repay = 0
            obs_t = self.env.reset()
            for _ in range(max_iter_per_epoch):
                try:
                    self.env.render()
                    time.sleep(0.02)
                except Exception as e:
                    pass
                act_t = sess.run(self.pi_op, feed_dict={self.obs_ph: obs_t.reshape(1, -1),
                                                        self.ou_noise_ph: np.zeros(self.env.act_dim).reshape(1, -1)})[0]
                rew_t, obs_n, don_n, _ = self.env.step(act_t)
                ep_repay += rew_t
                obs_t = obs_n

                if don_n:
                    self.logger.to_log('控制正常结束，总得分为：%.2f' % ep_repay)
                    break
            else:
                self.logger.to_log('控制达到最大控制次数，游戏被截断，总得分为：%.2f' % ep_repay)


