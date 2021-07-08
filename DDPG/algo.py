import tensorflow as tf
from tensorflow.train import AdamOptimizer
import numpy as np
import random
import os
from ENVS.AbstractEnv import AbsEnv
from .model import AbsControlModel, AbsEvaluateModel
from TOOLS.Logger import Logger
from TOOLS.noises import OrnsteinUhlenbeckActionNoise


class DataBuffer(object):
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        """
        数据缓存器的初始化

        :param obs_dim: int,
            状态观测向量的长度
        :param act_dim:  int,
            行动向量的长度
        :param size:  int,
            数据缓存器的总容量
        """
        # 存储当前的状态观测向量
        self.obs_t_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        # 存储当前的行动向量
        self.act_t_buffer = np.zeros((size, act_dim), dtype=np.float32)
        # 存储当前的奖励信号值
        self.reward_t_buffer = np.zeros((size, ), dtype=np.float32)
        # 存储衍生的状态观测向量
        self.obs_n_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        # 存储衍生的状态是否为终止状态
        self.done_n_buffer = np.zeros((size, ), dtype=np.float32)
        # 记录总数据条数的计数变量
        self.current_index = 0
        # 数据缓存器的大小
        self.size = size

    def store(self, obs_t: np.ndarray, act_t: np.ndarray, reward_t: float, obs_n: np.ndarray, done_n: bool) -> None:
        """
        决策器将观测到的一条数据记录写入到数据缓存器当中，这种写入会按照循环次序的方式进行

        :param obs_t: np.ndarray,
            决策器决策时所面临的状态
        :param act_t: np.ndarray,
            决策器基于当前所面临的状态所作出的决策--行动向量
        :param reward_t: float,
            决策器观测到的环境对象执行动作向量之后所得到的奖励信号数值
        :param obs_n: np.ndarray,
            决策器观测到的环境对象执行动作向量之后所得到的衍生状态观测向量
        :param done_n:
            决策器观测到的环境对象执行动作向量之后所得到的衍生状态观测是否是终止状态
        :return: None
            无返回值
        """
        # 当前插入样本的对应索引
        item_current_index = self.current_index % self.size
        # 向数据缓存器中插入当前记录
        self.obs_t_buffer[item_current_index] = obs_t
        self.act_t_buffer[item_current_index] = act_t
        self.reward_t_buffer[item_current_index] = reward_t
        self.obs_n_buffer[item_current_index] = obs_n
        self.done_n_buffer[item_current_index] = done_n
        # 总数据计数器加1
        self.current_index += 1

    def get(self, sample_size: int) -> tuple:
        """
        从数据缓存器中抽取出特定数量的样本组

        :param sample_size: int,
            抽取样本组的数量
        :return: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            返回各个子容器特定索引下的样本组
        """
        sample_index = random.sample(range(min(self.current_index, self.size)), sample_size)
        return self.obs_t_buffer[sample_index], self.act_t_buffer[sample_index], self.reward_t_buffer[sample_index], \
            self.obs_n_buffer[sample_index], self.done_n_buffer[sample_index]

    def get_buffer_size(self) -> int:
        """
        获取当前数据缓存器中的有效样本数量

        :return: int,
            返回当前数据缓存器中的有效样本数量
        """
        return min(self.current_index, self.size)


class DDPG(object):
    def __init__(self, env: AbsEnv, policy_model: AbsControlModel, evaluate_model: AbsEvaluateModel, save_dir: str,
                 exp_name: str, logger: Logger, gamma: float, eva_lr: float, pol_lr: float, rho: float,
                 conti_act_low: np.ndarray, conti_act_high: np.ndarray, is_OU_noise: bool = False):
        """
        深度确定性策略梯度方法（DDPG）是一种适用于连续控制指令场景下的决策机，其特点是策略网络给出的是确定行动，而不是行动概率分布，其缺点
        也是由于这个原因，策略网络本身没有办法进行探索，所以在训练过程中需要在策略网络输出结果上添加噪音。

        :param env: AbsEnv,
            与DDPG决策器进行交互的环境对象，这里使用的环境对象需要满足AbsEnv的接口规范
        :param policy_model: AbsControlModel,
            策略网络对象
        :param evaluate_model: AbsEvaluateModel,
            估值网络对象
        :param save_dir: str,
            策略网络对象和估值网络对象模型参数存储文件地址
        :param exp_name: str,
            本次试验的名称
        :param logger: Logger,
            日志对象
        :param gamma: float,
            远期奖励的折现系数
        :param eva_lr: float,
            估值网络的学习速率
        :param pol_lr: float,
            策略网络的学习速率
        :param rho: float,
            目标网络向主网络靠近的速率
        :param conti_act_low: np.ndarray,
            连续运动指令的下限
        :param conti_act_high: np.ndarray,
            连续运动指令的上限
        :param is_OU_noise: bool,
            控制过程中的随机噪音是否引入OU噪音
        """
        # 与决策器进行交互的环境对象
        self.env = env
        # 策略网络模型
        self.policy_model = policy_model
        # 估值网络模型
        self.evaluate_model = evaluate_model
        # 远期奖励的折现系数
        self.gamma = gamma
        # 估值网络的学习率
        self.evaluate_lr = eva_lr
        # 策略网络的学习率
        self.policy_lr = pol_lr
        # 目标网络向主网络的靠近速率
        self.rho = rho
        # 日志文件对象
        self.logger = logger
        # 模型参数存储文件夹
        self.target_dir_path = os.path.join(save_dir, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.mkdir(self.target_dir_path)
        # 模型参数存储文件名称
        self.target_file_name = os.path.join(self.target_dir_path, 'DDPG_Model.ckpt')
        # 连续动作指令的上限和下限
        self.conti_act_low = conti_act_low
        self.conti_act_high = conti_act_high
        # 控制学习过程中是否引入OU噪音
        self.is_OU_noise = is_OU_noise

        self.ph_reward, self.ph_done, self.ph_obs_new, self.ph_obs, self.ph_act, self.policy, self.target_policy,\
        self.evaluate_act, self.evaluate_policy, self.target_evaluate_policy, self.evaluate_loss, self.train_evaluate, \
        self.policy_loss, self.train_policy, self.update_target_params, self.init_target_params = self.build_model()

    def build_model(self) -> list:
        """
        定义一个TensorFlow计算图

        :return: tuple,
            返回一组用于DDPG更新的占位符和计算节点
        """
        #  ++++++++++++++++++ 占位符 ++++++++++++++++
        ph_reward = tf.placeholder(dtype=tf.float32, shape=(None, ), name='Reward_PH')
        ph_done = tf.placeholder(dtype=tf.float32, shape=(None, ), name='Done_PH')
        ph_obs_new = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='Obs_New_PH')
        ph_obs = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='Obs_PH')
        ph_act = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='Act_PH')
        #  ++++++++++++++++++ 占位符 ++++++++++++++++
        #  +++++++++++++++++ 模型定义 ++++++++++++++++
        main_policy_name = 'MainPolicy'
        main_evaluate_name = 'MainEvaluate'
        target_policy_name = 'TargetPolicy'
        target_evaluate_name = 'TargetEvaluate'
        with tf.variable_scope(main_policy_name):
            policy = self.policy_model.build_model(ph_obs, self.conti_act_high)
        with tf.variable_scope(target_policy_name):
            target_policy = self.policy_model.build_model(ph_obs_new, self.conti_act_high)
        with tf.variable_scope(main_evaluate_name):
            evaluate_act, evaluate_policy = self.evaluate_model.build_model(ph_obs, ph_act, policy)
        with tf.variable_scope(target_evaluate_name):
            _, target_evaluate_policy = self.evaluate_model.build_model(ph_obs_new, ph_act, target_policy)
        # 主策略网络的全体参数
        policy_params = [i for i in tf.global_variables() if main_policy_name in i.name]
        evaluate_params = [i for i in tf.global_variables() if main_evaluate_name in i.name]
        target_policy_params = [i for i in tf.global_variables() if target_policy_name in i.name]
        target_evaluate_params = [i for i in tf.global_variables() if target_evaluate_name in i.name]
        #  +++++++++++++++++ 模型定义 ++++++++++++++++
        #  +++++++++++++++++   算法  ++++++++++++++++
        # 混合了真实奖励信号的部分准确回报
        real_repay = ph_reward + self.gamma * (1 - ph_done) * target_evaluate_policy
        # 梯度下降在这一个分项上需要进行截断
        real_repay = tf.stop_gradient(real_repay)
        # 估值网络损失函数
        evaluate_loss = tf.reduce_mean(tf.square(evaluate_act - real_repay))
        train_evaluate = AdamOptimizer(learning_rate=self.evaluate_lr).minimize(evaluate_loss, var_list=evaluate_params)
        # 策略网络损失函数
        policy_loss = - tf.reduce_mean(evaluate_policy)
        train_policy = AdamOptimizer(learning_rate=self.policy_lr).minimize(policy_loss, var_list=policy_params)
        # 更新目标网络参数
        update_target_policy_params = tf.group([tf.assign(param_targ, self.rho * param_main + (1-self.rho) * param_targ)
                                                for param_main, param_targ in zip(policy_params, target_policy_params)])
        update_target_evaluate_params = tf.group([tf.assign(param_targ, self.rho * param_main + (1-self.rho) * param_targ)
                                                  for param_main, param_targ in zip(evaluate_params,
                                                                                    target_evaluate_params)])
        update_target_params = tf.group([update_target_policy_params, update_target_evaluate_params])
        # 初始化目标网络的参数等于主网络的参数
        init_target_policy_params = tf.group([tf.assign(param_targ, param_main)
                                              for param_main, param_targ in zip(policy_params, target_policy_params)])
        init_target_evaluate_params = tf.group([tf.assign(param_targ, param_main)
                                                for param_main, param_targ in zip(evaluate_params,
                                                                                  target_evaluate_params)])
        init_target_params = tf.group([init_target_policy_params, init_target_evaluate_params])
        #  +++++++++++++++++   算法  ++++++++++++++++
        return [ph_reward, ph_done, ph_obs_new, ph_obs, ph_act, policy, target_policy, evaluate_act,
                evaluate_policy, target_evaluate_policy, evaluate_loss, train_evaluate, policy_loss,
                train_policy, update_target_params, init_target_params]

    def train(self, buffer_size: int = 1000000, retrain_label: bool = False, learn_epochs: int = 150,
              max_iter_per_epoch: int = 4000, sample_size: int = 200, save_freq: int = 10, noise_scale: float = 0.1,
              start_steps: int = 10000, update_after: int = 1000, update_every: int = 50, render: bool = False):
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型存储对象
        model_saver = tf.train.Saver(max_to_keep=5)
        # 创建数据缓存器
        data_buffer = DataBuffer(self.env.obs_dim, self.env.act_dim, buffer_size)
        # 如果是重新进行训练的话，就读取历史参数记录
        if retrain_label:
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        # 否则的话，就初始化计算资源会话中的各个变量
        else:
            sess.run(tf.global_variables_initializer())
            # 同时将目标网络的参数初始化为主网络中的参数
            _ = sess.run(self.init_target_params)
        # 开始进行测试和训练
        def get_action(obs_t, noise_scale):
            act = sess.run(self.policy, feed_dict={self.ph_obs: obs_t.reshape(1, -1)})
            # 目前只有一个状态观测向量，所以只衍生出一个行动向量
            act = act[0]
            # 在行动向量上添加随机噪音
            if not self.is_OU_noise:
                act += noise_scale * np.random.randn(self.env.act_dim)
            else:
                act += OU_noise()
            return np.clip(act, self.conti_act_low, self.conti_act_high)
        # 总决策次数等于单回合最大决策次数乘以学习回合总数
        total_steps = max_iter_per_epoch * learn_epochs
        obs_t, ep_repay, ep_len = self.env.reset(), 0, 0
        if self.is_OU_noise:
            OU_noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.env.act_dim, ), sigma=0.75)
        k = 1
        for t in range(total_steps):
            if render:
                try:
                    self.env.render()
                except Exception as e:
                    pass

            if t > start_steps:
                act = get_action(obs_t, noise_scale)
            else:
                if not self.is_OU_noise:
                    act = self.conti_act_high * (np.random.rand(self.env.act_dim) * 2 - 1)
                else:
                    act = OU_noise()
            reward, obs_n, done, _ = self.env.step(act)
            ep_repay += reward
            ep_len += 1
            done = False if ep_len == max_iter_per_epoch else done
            data_buffer.store(obs_t, act, reward, obs_n, done)
            obs_t = obs_n

            if done or (ep_len == max_iter_per_epoch):
                self.logger.to_log('[%d]控制回合结束！总得分为：%.2f' % (k, ep_repay))
                obs_t, ep_repay, ep_len = self.env.reset(), 0, 0
                k += 1
                # OU噪音重置为初始状态
                OU_noise.reset()

            if t >= update_after and t % update_every == 0:
                evaluate_loss_list = []
                policy_loss_list = []
                for _ in range(update_every):
                    obs_t_buf, act_t_buf, rew_t_buf, obs_n_buf, done_buf = data_buffer.get(sample_size)
                    feed_dict = {self.ph_obs: obs_t_buf, self.ph_act: act_t_buf, self.ph_reward: rew_t_buf,
                                 self.ph_obs_new: obs_n_buf, self.ph_done: done_buf}
                    evaluate_outs = sess.run([self.evaluate_loss, self.train_evaluate], feed_dict=feed_dict)
                    policy_outs = sess.run([self.policy_loss, self.train_policy], feed_dict=feed_dict)
                    if np.random.rand() < 0.3:
                        sess.run(self.update_target_params)
                    evaluate_loss_list.append(evaluate_outs[0])
                    policy_loss_list.append(policy_outs[0])
                mean_evaluate_loss = np.mean(evaluate_loss_list)
                mean_policy_loss = np.mean(policy_loss_list)
                if t % 4000 == 0:
                    self.logger.to_log('估值网络上的损失为：%.6f' % mean_evaluate_loss)
                    self.logger.to_log('策略网络上的得分为：%.6f' % (-mean_policy_loss))
            if t % save_freq == 0:
                model_saver.save(sess, self.target_file_name, global_step=t)

        # 关闭计算资源
        sess.close()

    def test(self, test_epochs, max_iter_per_epoch):
        import time
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型存储对象
        model_saver = tf.train.Saver()
        # 向计算资源会话中恢复模型参数
        model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))

        for epoch in range(test_epochs):
            self.logger.to_log('+++++++ Excuting - [%d] +++++++' % epoch)
            obs_t = self.env.reset()
            ep_repay = 0
            for k in range(max_iter_per_epoch):
                try:
                    self.env.render()
                    time.sleep(0.02)
                except Exception:
                    pass

                act_t = sess.run(self.policy, feed_dict={self.ph_obs: obs_t.reshape(1, -1)})
                reward_t, obs_t, done, _ = self.env.step(act_t[0])
                ep_repay += reward_t
                if done or k == max_iter_per_epoch - 1:
                    self.logger.to_log('控制回合结束，总得分为：%.2f' % ep_repay)
                    ep_repay = 0
                    break

