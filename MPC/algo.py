import tensorflow as tf
import numpy as np
from ENVS.AbstractEnv import AbsEnv
from TOOLS.Logger import Logger
from .model import AbsControlModel, AbsEvaluateModel
import os, math
import random, time
import matplotlib.pyplot as plt


class MPC(object):
    def __init__(self, env: AbsEnv, policy_model: AbsControlModel, evaluate_model: AbsEvaluateModel,
                 exp_name: str, clip_ratio: float, lr: float = 0.0001, model_save_dir: str = 'MODEL_PARAMS'):
        self.env = env
        self.learning_rate = lr
        self.policy_model = policy_model
        self.evaluate_model = evaluate_model
        self.target_dir_path = os.path.join(model_save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        self.target_file_path = os.path.join(self.target_dir_path, 'MPC_Model.ckpt')
        self.clip_ratio = clip_ratio

        self.ph_obs_t, self.ph_act_t, self.ph_adv_t, self.ph_rep_t, self.act_t, self.val, self.update_policy_params, \
            self.adv_loss, self.train_pol_op, self.eva_loss, self.train_eva_op = self.build_model()

    def build_model(self):
        # ++++++++ 占位符 ++++++++
        ph_obs_t = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='PH_Obs_t')
        ph_act_t = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='PH_Act_t')
        ph_adv_t = tf.placeholder(dtype=tf.float32, shape=(None, ), name='PH_Adv_t')
        ph_rep_t = tf.placeholder(dtype=tf.float32, shape=(None, ), name='PH_Rep_t')
        # ++++++++ 占位符 ++++++++
        # ++++++++ 模型 ++++++++++
        main_policy_model_name = 'MainPolicyModel'
        pre_policy_model_name = 'PrePolicyModel'
        evaluate_model_name = 'EvaluateModel'
        with tf.variable_scope(main_policy_model_name):
            main_pi, act_t = self.policy_model.build_model(ph_obs_t)
        with tf.variable_scope(pre_policy_model_name):
            pre_pi, _ = self.policy_model.build_model(ph_obs_t)
        with tf.variable_scope(evaluate_model_name):
            val = self.evaluate_model.build_model(ph_obs_t)
        main_policy_params = [i for i in tf.global_variables() if main_policy_model_name in i.name]
        pre_policy_params = [i for i in tf.global_variables() if pre_policy_model_name in i.name]
        evaluate_params = [i for i in tf.global_variables() if evaluate_model_name in i.name]
        update_policy_params = [param_pre.assign(param_main)
                                for param_main, param_pre in zip(main_policy_params, pre_policy_params)]
        # ++++++++ 模型 ++++++++++
        # ++++++++ 算法 ++++++++++
        prob_ph_act_t = main_pi.prob(ph_act_t)
        pre_prob_ph_act_t = tf.stop_gradient(pre_pi.prob(ph_act_t))
        # 重要性采样比率
        ratio = prob_ph_act_t / pre_prob_ph_act_t
        adv_gain = ratio * ph_adv_t
        adv_loss = -tf.reduce_mean(tf.minimum(adv_gain,
                                              tf.clip_by_value(ratio, 1.-self.clip_ratio, 1.+self.clip_ratio)*ph_adv_t))
        train_pol_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(adv_loss)
        eva_loss = tf.reduce_mean(tf.square(ph_rep_t - val))
        train_eva_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(eva_loss)

        return ph_obs_t, ph_act_t, ph_adv_t, ph_rep_t, act_t, val, update_policy_params, adv_loss, train_pol_op, \
            eva_loss, train_eva_op

    def train(self, learning_epoch, max_iter_per_epoch, retrain_label):
        sess = tf.Session()
        model_saver = tf.train.Saver(max_to_keep=5)
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(learning_epoch):
            pass

    def get_act_t(self, obs_t):
        act_t = self.sess.run(self.act_t, feed_dict={self.ph_obs_t: obs_t.reshape(1, -1)})[0]
        return act_t

    def init_session(self):
        self.sess = tf.Session()
        model_saver = tf.train.Saver(max_to_keep=5)
        try:
            model_saver.restore(self.sess, tf.train.latest_checkpoint(self.target_dir_path))
        except:
            self.sess.run(tf.global_variables_initializer())


class Sampler(object):
    def __init__(self, env: AbsEnv, actor, gamma: float, act_lim: np.ndarray,
                 max_iter_per_epoch: int):
        self.env = env
        self.actor = actor
        try:
            self.actor.init_session()
        except Exception as e:
            print(e)
            pass
        self.gamma = gamma
        self.act_lim = act_lim
        self.max_iter_per_epoch = max_iter_per_epoch

    def gene_mean_std(self, epoch_num):
        batch_obs_t = []
        batch_act_t = []
        batch_obs_n = []
        batch_rew_t = []
        for epoch in range(epoch_num):
            print('.', end='')
            obs_t = self.env.reset()
            for _ in range(self.max_iter_per_epoch):
                # 后续行动的决策和环境状态无关，所以，这一状态观测向量不参与计算
                obs_t_reshaped = obs_t.reshape((1, -1))
                # 随机动作
                random_act_t = (np.random.randn(len(self.act_lim)) - 0.5) * self.act_lim
                rew_t, obs_n, don_n, _ = self.env.step(random_act_t)
                batch_obs_t.append(obs_t)
                batch_act_t.append(random_act_t)
                batch_obs_n.append(obs_n)
                batch_rew_t.append((rew_t + 8) / 8)
                if don_n:
                    break
                obs_t = obs_n
        batch_obs_t = np.array(batch_obs_t)
        batch_act_t = np.array(batch_act_t)
        batch_obs_n = np.array(batch_obs_n)
        batch_rew_t = np.array(batch_rew_t)
        batch_delta_obs = batch_obs_n - batch_obs_t
        obs_mean, obs_std = self.normalize(batch_obs_t)
        act_mean, act_std = self.normalize(batch_act_t)
        self.delta_obs_mean, self.delta_obs_std = self.normalize(batch_delta_obs)
        self.obs_act_mean = np.hstack((obs_mean, act_mean))
        self.obs_act_std = np.hstack((obs_std, act_std))

    def sample_paths(self, epoch_num, render=False):
        batch_obs_t = []
        batch_act_t = []
        batch_obs_n = []
        batch_rew_t = []
        for epoch in range(epoch_num):
            print('[%d]开始测试' % epoch)
            obs_t = self.env.reset()
            exp_repay = 0
            for epoch_index in range(self.max_iter_per_epoch):
                if render:
                    self.env.render()
                    time.sleep(0.03)
                act_t = self.actor.get_act_t(obs_t)
                rew_t, obs_n, don_n, _ = self.env.step(act_t)
                exp_repay += rew_t
                batch_obs_t.append(obs_t)
                batch_act_t.append(act_t)
                batch_obs_n.append(obs_n)
                batch_rew_t.append((rew_t + 8) / 8)
                if don_n:
                    print('游戏正常结束，总得分为：%.2f' % exp_repay)
                    break
                obs_t = obs_n
            else:
                print('游戏达到最大控制次数，总得分为：%.2f' % exp_repay)
        batch_obs_t = np.array(batch_obs_t)
        batch_act_t = np.array(batch_act_t)
        batch_obs_n = np.array(batch_obs_n)
        batch_rew_t = np.array(batch_rew_t)
        batch_delta_obs = batch_obs_n - batch_obs_t
        batch_obs_act_t = np.hstack((batch_obs_t, batch_act_t))
        return batch_obs_act_t, batch_delta_obs, batch_obs_n

    def normalize(self, batch_data):
        mean = np.mean(batch_data, 0)
        std = np.std(batch_data, 0)
        return mean, std


class EnvironmentModel(object):
    def __init__(self, sampler: Sampler, obs_dim, act_dim, learning_rate, model_save_dir,
                 exp_name, batch_size, logger: Logger):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.learning_rate = learning_rate
        self.sampler = sampler
        self.sampler.gene_mean_std(100)
        self.batch_size = batch_size
        self.target_dir_path = os.path.join(model_save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        self.target_file_path = os.path.join(self.target_dir_path, 'MPC_Env_Model.ckpt')
        self.logger = logger

        self.ph_obs_act_t, self.ph_delta_obs, self.predict, self.loss, self.train_dynamic_op = self.build_model()

    def dynamic_model(self, ph_obs_act_t):
        f_1 = tf.layers.dense(inputs=ph_obs_act_t, units=200, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                   bias_initializer=tf.constant_initializer(0.1))
        f_2 = tf.layers.dense(inputs=f_1, units=100, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                   bias_initializer=tf.constant_initializer(0.1))
        predict = tf.layers.dense(inputs=f_2, units=self.obs_dim)
        return predict

    def build_model(self):
        # ++++++++++++++ 占位符 +++++++++++++++++++
        ph_obs_act_t = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim+self.act_dim),
                                           name='PH_Obs_Act_t')
        ph_delta_obs = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
        # ++++++++++++++ 占位符 +++++++++++++++++++
        # ++++++++++++++ 模型 +++++++++++++++++++
        predict = self.dynamic_model(ph_obs_act_t)
        loss = tf.reduce_mean(tf.square(predict - ph_delta_obs))
        train_dynamic_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # ++++++++++++++ 模型 +++++++++++++++++++
        return ph_obs_act_t, ph_delta_obs, predict, loss, train_dynamic_op

    def train(self, retrain_label, train_num):
        sess = tf.Session()
        model_saver = tf.train.Saver(max_to_keep=5)
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())
        batch_obs_act_t, batch_delta_obs, _ = self.sampler.sample_paths(200)
        train_obs_act_t = (batch_obs_act_t - self.sampler.obs_act_mean) / self.sampler.obs_act_std
        train_delta_obs = (batch_delta_obs - self.sampler.delta_obs_mean) / self.sampler.delta_obs_std
        sample_num = train_delta_obs.shape[0]
        train_indicies = np.arange(sample_num)
        smoothed_loss = 0
        stop_flag = 0
        for i in range(train_num):
            np.random.shuffle(train_indicies)
            for j in range(int(math.ceil(sample_num / self.batch_size))):
                start_idx = j * self.batch_size % sample_num
                idx = train_indicies[start_idx: start_idx+self.batch_size]
                sess.run(self.train_dynamic_op, feed_dict={self.ph_obs_act_t: train_obs_act_t[idx, :],
                                                           self.ph_delta_obs: train_delta_obs[idx, :]})
                loss = sess.run(self.loss, feed_dict={self.ph_obs_act_t: train_obs_act_t[idx, :],
                                                      self.ph_delta_obs: train_delta_obs[idx, :]})
                if i == 0:
                    smoothed_loss = loss
                else:
                    smoothed_loss = 0.95 * smoothed_loss + 0.05 * loss
                if smoothed_loss < 0.0001:
                    stop_flag = 1
                    break
            if stop_flag == 1:
                break
            print('第%d次试验之后，误差为：%.6f' % (i, smoothed_loss))
        self.logger.to_log('存储模型参数')
        model_saver.save(sess, self.target_file_path, global_step=train_num)

    def predict_func(self, obs_act_t, obs_n=None):
        if len(obs_act_t.shape) == 1:
            obs_act_t = obs_act_t.reshape(1, -1)
        try:
            sess = self.sess
        except:
            sess = tf.Session()
            model_saver = tf.train.Saver(max_to_keep=5)
            try:
                model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
            except:
                sess.run(tf.global_variables_initializer())
        normed_obs_act_t = (obs_act_t-self.sampler.obs_act_mean) / self.sampler.obs_act_std
        delta_obs = sess.run(self.predict, feed_dict={self.ph_obs_act_t: normed_obs_act_t})
        pred_obs_n = delta_obs * self.sampler.delta_obs_std + self.sampler.delta_obs_mean + obs_act_t[:, 0:3]
        xs = np.arange(len(pred_obs_n))
        if obs_n:
            figure = plt.figure()
            for index in [1, 2, 3]:
                ax_1 = figure.add_subplot(3,1, index)
                plt.plot(xs, pred_obs_n[:, index-1], color='r', ls='--')
                plt.plot(xs, obs_n[:, index-1], color='k', lw=1.4, alpha=0.65)
            plt.show()
        return pred_obs_n

    def init_session(self, read_params):
        self.sess = tf.Session()
        model_saver = tf.train.Saver(max_to_keep=5)
        if read_params:
            model_saver.restore(self.sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            self.sess.run(tf.global_variables_initializer())


class RandomMPCController(object):
    def __init__(self, env_model, act_lim, obs_dim, act_dim, depth=20, num_simulated_paths=200):
        # 环境模型
        self.env_model = env_model
        # 当前环境模型所能接受的控制指令上限
        self.act_lim = act_lim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # 模拟控制的深度
        self.depth = depth
        # 并行模拟控制的数量
        self.num_simulated_path = num_simulated_paths

    def get_act_t(self, obs_t):
        # 模拟状态切片，模拟状态-行动序列
        obs_layer, obs_act_seq = [], []
        # 将初始的状态填写到并行模拟状态序列的起始位置
        for _ in range(self.num_simulated_path):
            obs_layer.append(obs_t)
        # 在每一次迭代深度上
        for d in range(self.depth):
            # 添加这一层深度上模拟动作，目前都是随机动作
            act_layer = []
            for _ in range(self.num_simulated_path):
                act_layer.append((np.random.rand(len(self.act_lim))-0.5) * self.act_lim)
            #
            obs_act_layer = np.hstack((np.array(obs_layer), np.array(act_layer)))
            obs_act_seq.append(obs_act_layer)
            obs_n_layer = self.env_model.predict_func(obs_act_layer)
            obs_layer = obs_n_layer
        rew_accum = self.compute_rew(obs_act_seq)
        optimal_index = np.argmax(rew_accum)
        return obs_act_seq[0][optimal_index, self.obs_dim:self.obs_dim+self.act_dim]

    def compute_rew(self, obs_act_seq):
        rew_accum = np.zeros(self.num_simulated_path)
        for i in range(self.num_simulated_path):
            for j in range(self.depth):
                rew_accum[i] += -(math.atan2(obs_act_seq[j][i, 1], obs_act_seq[j][i, 0])**2
                                  + .1*obs_act_seq[j][i, 2]**2 + 0.001*obs_act_seq[j][i, 3]**2)
        return rew_accum

def test():
    from ENVS.Envs import PendulumEnv
    from .model import MLPContiControlModel, MLPEvaluateModel
    from TOOLS.Logger import LoggerPrinter

    logger = LoggerPrinter()
    env = PendulumEnv(logger)
    act_lim = np.array([2.0, ])
    policy_model = MLPContiControlModel(env.obs_dim, env.act_dim, hidden_sizes=(20, 10), hd_activation='ReLU',
                                        act_lim=np.array([2.0, ]), logger=logger)
    evaluate_model = MLPEvaluateModel(env.obs_dim, hidden_size=(20, 10), hd_activation='ReLU', logger=logger)
    mpc = MPC(env=env, policy_model=policy_model, evaluate_model=evaluate_model, exp_name='Pendulum', clip_ratio=0.1,
              lr=0.0001, model_save_dir="MODEL_PARAMS")
    sampler = Sampler(env=env, actor=mpc, gamma=0.9, act_lim=np.array([2.0, ]), max_iter_per_epoch=200)
    environment_model = EnvironmentModel(sampler=sampler, obs_dim=env.obs_dim, act_dim=env.act_dim,
                                         learning_rate=0.001, model_save_dir='MODEL_PARAMS', exp_name='Pendulum',
                                         batch_size=200, logger=logger)
    environment_model.train(retrain_label=False, train_num=200)
    environment_model.init_session(read_params=True)
    random_MPCor = RandomMPCController(env_model=environment_model, act_lim=act_lim, obs_dim=3, act_dim=1)
    sampler_2 = Sampler(env=env, actor=random_MPCor, gamma=0.9, act_lim=act_lim, max_iter_per_epoch=200)
    sampler_2.sample_paths(20, render=True)
    return sampler_2
