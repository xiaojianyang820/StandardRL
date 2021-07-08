import tensorflow as tf
import numpy as np
from tensorflow.train import AdamOptimizer
from scipy import signal
import os

# 除零修正系数
EPS = 1e-8


class GAEBuffer(object):
    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma, beta):
        self.obs_buffer = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buffer = np.zeros([size, act_dim], dtype=np.float32)
        self.val_buffer = np.zeros((size, ), dtype=np.float32)
        self.rew_buffer = np.zeros((size, ), dtype=np.float32)
        self.repay_buffer = np.zeros((size, ), dtype=np.float32)
        self.adv_buffer = np.zeros((size, ), dtype=np.float32)
        self.logp_act_buffer = np.zeros((size, ), dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + [v]) for k, v in info_shapes.items()}
        self.sorted_info_keys = self.info_bufs.keys()
        self.gamma = gamma
        self.beta = beta

        self.current_index, self.con_start_index, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        assert self.current_index < self.max_size
        self.obs_buffer[self.current_index] = obs
        self.act_buffer[self.current_index] = act
        self.rew_buffer[self.current_index] = rew
        self.val_buffer[self.current_index] = val
        self.logp_act_buffer[self.current_index] = logp

        for k in self.sorted_info_keys:
            self.info_bufs[k][self.current_index] = info[k]

        self.current_index += 1

    def discount_cumsum(self, x, discount):
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        path_slice = slice(self.con_start_index, self.current_index)
        rews = np.append(self.rew_buffer[path_slice], last_val)
        vals = np.append(self.val_buffer[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buffer[path_slice] = self.discount_cumsum(deltas, self.gamma * self.beta)
        self.repay_buffer[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.con_start_index = self.current_index

    def get(self):
        assert self.con_start_index == self.max_size
        self.current_index, self.con_start_index = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buffer), np.std(self.adv_buffer)
        self.adv_buffer = (self.adv_buffer - adv_mean) / adv_std
        return [self.obs_buffer, self.act_buffer, self.adv_buffer, self.repay_buffer, self.logp_act_buffer] + \
            [self.info_bufs[k] for k in self.sorted_info_keys]


class TRPO(object):
    def __init__(self, env, policy_model, evaluate_model, model_save_path, exp_name, logger,
                 gamma, beta, damping_coef, info_dict_keys, max_size_per_epoch, delta=0.01,
                 eva_lr=0.001, algo='trpo'):
        self.env = env
        self.policy_model = policy_model
        self.evaluate_model = evaluate_model
        self.target_dir_path = os.path.join(model_save_path, exp_name)
        if not os.path.exists(self.target_dir_path):
            os.makedirs(self.target_dir_path)
        self.target_file_name = os.path.join(self.target_dir_path, 'TRPO_Model.cpkt')
        self.logger = logger
        self.gamma, self.beta = gamma, beta
        self.damping_coef = damping_coef
        self.delta = delta
        self.info_dict_keys = info_dict_keys
        self.eva_lr = eva_lr
        self.max_size_per_epoch = max_size_per_epoch
        self.algo = algo

        self.H_fxx, self.x_vec_ph, self.obs_ph, self.info_dict_ph, self.gradient, self.pol_loss,\
            self.eva_loss, self.adv_ph, self.logp_act_old_ph, self.act_ph, self.repay_ph,\
            self.get_policy_params, self.set_policy_params, self.kl_divergence, \
            self.train_evaluate, self.pi, self.value, self.logp_pi, self.info_dict = self.build_model()

    def build_model(self):
        # ++++++++++++++++++++++ 定义占位符 ++++++++++++++++++++++++++++++++
        # 即刻状态观测向量的占位符
        obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.obs_dim), name='Obs_PH')
        # 即刻行动向量的占位符
        act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.act_dim), name='Act_PH')
        # 即刻决策分布参量的占位符
        info_dict_ph = {}
        for k, shape in self.info_dict_keys.items():
            info_dict_ph[k] = tf.placeholder(dtype=tf.float32, shape=(None, shape), name='Info_%s_PH' % k)
        # 在当前策略网络参数下即刻行动向量的对数似然
        logp_act_old_ph = tf.placeholder(dtype=tf.float32, shape=(None, ), name='Logp_Act_Old_PH')
        # 即刻行动向量的广义优势估计占位符
        adv_ph = tf.placeholder(dtype=tf.float32, shape=(None, ), name='Adv_PH')
        # 即刻状态观测向量的实际价值估算占位符
        repay_ph = tf.placeholder(dtype=tf.float32, shape=(None, ), name='Repay_PH')
        # ++++++++++++++++++++++ 定义占位符 ++++++++++++++++++++++++++++++++
        # ++++++++++++++++++++++ 构建策略网络和估值网络接口 +++++++++++++++++++
        with tf.variable_scope('Policy'):
            # pi: 策略网络基于状态观测向量计算出决策分布，然后从该分布中进行抽样得到的行动向量
            # logp_pi: 本次采样得到的行动向量对应的对数似然
            # logp_act_ph: 当前策略网络对于给定行动向量的对数似然
            # info_dict: 策略网络基于状态观测向量计算出来的决策分布的参数信息
            # kl_divergence: 给定状态观测向量，原有的决策分布参数信息，计算出更新的策略网络在当前状态观测向量上与之前的策略网络之间
            #                的KL散度
            pi, logp_pi, logp_act_ph, info_dict, kl_divergence = self.policy_model.build_model(
                obs_ph, act_ph, info_dict_ph
            )
        with tf.variable_scope('Evaluate'):
            # value: 估值网络对于状态观测向量的价值评估
            value = self.evaluate_model.build_model(obs_ph)
        # ++++++++++++++++++++++ 构建策略网络和估值网络接口 +++++++++++++++++++
        # ++++++++++++++++++++++ TRPO算法 +++++++++++++++++++++++++++++++++
        # 定义策略网络的损失函数
        #   策略重要性采样比率
        ratio = tf.exp(logp_act_ph - logp_act_old_ph)
        #   策略广义优势函数的期望（参数的变化会引起某一行动出现概率的变化，这种变化如果和这一行动的广义优势是同一方向的，
        #   那么就显然可以促进策略得分的上升）
        pol_loss = -tf.reduce_mean(ratio * adv_ph)
        # 定义估值网络的损失函数
        evaluate_loss = tf.reduce_mean(tf.square(repay_ph - value))
        train_evaluate = AdamOptimizer(learning_rate=self.eva_lr).minimize(evaluate_loss)
        # 获取策略网络的全部待估参数
        scope = 'Policy'
        policy_params = [x for x in tf.trainable_variables() if scope in x.name]
        # 策略网络损失函数对所有的可训练参数进行求导
        gradient = tf.gradients(xs=policy_params, ys=pol_loss)
        # 经过上一步求导之后，得到的梯度对象实际上是不同形状的矩阵所构成的元组
        # 为了方便后续计算，按照固定顺序，将这些矩阵平坦化，然后拼接为一个向量
        gradient = tf.concat([tf.reshape(i, (-1, )) for i in gradient], axis=0)

        # 这里实现了一个可以快速计算某一函数f的海森矩阵与某一个向量x相乘所得到的结果向量
        # 该函数的参数f代表这一个特定的函数，而参数params代表的是该函数中的可微分参数
        def hessian_vector_product(f, params):
            # 由于这里的函数f都是由神经网络所表示的，所以参数都是以矩阵的形式存在
            # 故求导之后依然是一个矩阵matrix
            gm = tf.gradients(f, params)
            # 将梯度矩阵平坦化，然后拼接为一个向量
            g = tf.concat([tf.reshape(x, (-1, )) for x in gm], axis=0)
            # 目标向量x的占位符
            x = tf.placeholder(dtype=tf.float32, shape=g.shape)
            # 梯度向量点乘目标向量之后再次对变量进行梯度运算，就可以得到原来函数的海森矩阵与目标向量叉乘的结果
            gm_fxx = tf.gradients(tf.reduce_sum(g * x), params)
            # 将上述结果再次平坦化和拼接，构成一个向量
            H_fxx = tf.concat([tf.reshape(i, (-1, )) for i in gm_fxx], axis=0)
            # 返回目标向量的占位符和该结果的计算节点
            return x, H_fxx
        x_vec_ph, H_fxx = hessian_vector_product(kl_divergence, policy_params)
        # damping_coef是一个稳健系数，主要是为了避免Hessian矩阵为退化阵
        if self.damping_coef > 0:
            H_fxx += self.damping_coef * x_vec_ph
        # 获取策略网络的全部参数，整理为一个向量
        get_policy_params = tf.concat([tf.reshape(i, (-1, )) for i in policy_params], axis=0)
        # x_vec_ph占位符可以作为更新参数向量的注入位置，不过需要进一步将这个占位符中的数值整理成和
        # policy_params一个的形状
        flat_size = lambda p: int(np.prod(p.shape.as_list()))
        splits = tf.split(x_vec_ph, [flat_size(p) for p in policy_params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(policy_params, splits)]
        set_policy_params = tf.group([tf.assign(p, p_new) for p, p_new in zip(policy_params, new_params)])
        # ++++++++++++++++++++++ TRPO算法 +++++++++++++++++++++++++++++++++
        evaluate_params = [x for x in tf.trainable_variables() if 'Evaluate' in x.name]

        return H_fxx, x_vec_ph, obs_ph, info_dict_ph, gradient, pol_loss, evaluate_loss, adv_ph, logp_act_old_ph,\
               act_ph, repay_ph, get_policy_params, set_policy_params, kl_divergence, train_evaluate, pi, value,\
               logp_pi, info_dict

    def train(self, retrain_label, cg_iters, backtrack_iters, backtrack_coef, train_v_iters, learn_epochs,
              threshold, max_con_len, save_freq=10):
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型存储器
        model_saver = tf.train.Saver()
        # 如果是重启训练的话，就加载原有的模型参数，否则的话，就初始化计算图中的全部参数
        if retrain_label:
            self.logger.to_log('开始恢复模型参数')
            model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        else:
            sess.run(tf.global_variables_initializer())
        # 创建一个数据缓存器
        buffer = GAEBuffer(self.env.obs_dim, self.env.act_dim, size=self.max_size_per_epoch,
                           gamma=self.gamma, beta=self.beta, info_shapes=self.info_dict_keys)

        # 定义共轭梯度法，用于求解在特定策略网络参数矩阵条件下能够满足Hx=g的目标向量x
        # 这里的H是策略网络对于的KL散度函数的Hessian矩阵，g是广义优势损失的雅克比向量
        # 原目标是求解一个目标向量x=H(-1)g，但是由于求解KL散度函数海森矩阵的逆是时间复杂度是O(3),
        # 所以引入共轭梯度的方式来求解目标向量x，这一方法的时间复杂度大约为O(2)
        def cg(Ax, b, cg_iters):
            # 最开始的目标向量x是一个零向量，形状和梯度向量b相同
            x = np.zeros_like(b)
            # 总残差向量r，由于当前的目标向量x为零向量，所以残差向量r=b-Ax等于向量b
            r = b.copy()
            # 共轭分量d，第一个共轭分量的方向选定和残差向量r相同
            d = r.copy()
            # 总残差向量的摸
            r_dot_old = np.dot(r, r)
            for _ in range(cg_iters):
                # 当前的共轭分量经过Hessian矩阵投影之后得到的分量
                z = Ax(d)
                # 投影前的向量d和投影后的向量z的内积反映了向量d和投影矩阵H的相似程度
                # 用目前的残差向量的模去除以向量d投影之后丢失的长度，得到一个放缩系数alpha
                alpha = r_dot_old / (np.dot(d, z) + EPS)
                # 共轭分量d按照alpha的比例添加到目标向量x中
                x += alpha * d
                # 从总残差向量中扣除已经得到解释的部分
                r -= alpha * z
                # 计算新的残差向量的模
                r_dot_new = np.dot(r, r)
                # 在总残差向量的基础上调整得到下一个共轭分量，调整的方向是向当前的共轭分量方向调整
                # 调整的力度是剩余残差的比率
                d = r + (r_dot_new / r_dot_old) * d
                r_dot_old = r_dot_new
            return x

        def update():
            self.logger.to_log('对策略网络进行更新训练')
            # 读取历史数据
            batch_data = buffer.get()
            # b_obs: 状态观测数据组        # b_adv: 广义优势数据组
            # b_act: 行动数据组           # b_repay: 状态真实价值估计数据组
            # b_logp_act: 行动对数似然数据组
            b_obs, b_act, b_adv, b_repay, b_logp_act = batch_data[:5]
            # 索引5之后的数据组是刻画决策分布的数据组
            b_info_list = batch_data[5:]

            # 构建一个供共轭梯度算法使用的计算函数，可以快速计算Hessian矩阵和任意向量x之间的乘积
            # 这一计算依赖于三个不同的占位符，1. x_vec_ph：目标向量x；2. obs_ph：状态观测数据组
            # 3. info_dict_ph：前置分布参数数据组
            # 基于状态观测可以计算出目前对应的分布参数（例如均值和标准差），再提供前置的分布参数（例如均值和标准差）
            # 就可以计算出KL散度函数，从而进一步计算出它的Hessian矩阵与任意目标向量x的叉乘
            def H_fx(x):
                here_feed_dict = {self.x_vec_ph: x, self.obs_ph: b_obs}
                for i, k in enumerate(self.info_dict_keys):
                    here_feed_dict[self.info_dict_ph[k]] = b_info_list[i]
                return sess.run(self.H_fxx, feed_dict=here_feed_dict)

            # 计算出广义策略损失对参数向量的梯度，广义策略损失的当前值，估值网络损失的当前值
            here_feed_dict = {self.adv_ph: b_adv, self.logp_act_old_ph: b_logp_act,
                              self.act_ph: b_act, self.obs_ph: b_obs, self.repay_ph: b_repay}
            g, pol_loss_old, eva_loss_old = sess.run([self.gradient, self.pol_loss, self.eva_loss],
                                                     feed_dict=here_feed_dict)
            self.logger.to_log('前置策略网络的广义优势损失为：%.4f; 前置估值网络的估计损失为：%.4f' % (pol_loss_old,
                                                                           eva_loss_old))
            # 利用共轭梯度法求解出目标向量x，满足Hx=g
            target_x = cg(H_fx, g, cg_iters)
            # 计算出更新因子alpha
            alpha = np.sqrt(2 * self.delta / (np.dot(target_x, H_fx(target_x)) + EPS))
            # 保存当前策略网络的网络参数
            old_params = sess.run(self.get_policy_params)

            def set_and_eval_policy_params(step):
                here_feed_dict = {self.x_vec_ph: old_params - alpha * step * target_x}
                # 完成对策略网络参数的更新
                sess.run(self.set_policy_params, feed_dict=here_feed_dict)
                # 计算出在新的参数下新策略与前置策略之间的KL散度和新参数下的广义策略损失
                here_feed_dict = {self.obs_ph: b_obs, self.act_ph: b_act, self.logp_act_old_ph: b_logp_act,
                                  self.adv_ph: b_adv}
                for i, k in enumerate(self.info_dict_keys):
                    here_feed_dict[self.info_dict_ph[k]] = b_info_list[i]
                return sess.run([self.kl_divergence, self.pol_loss], feed_dict=here_feed_dict)

            if self.algo == 'npg':
                # 如果使用的算法是NPG，那么就直接将更新步长设置为1
                update_kl_divergence, pol_loss_new = set_and_eval_policy_params(1.0)
            elif self.algo == 'trpo':
                # 如果使用的算法是TRPO，那么就需要使用线搜索的方法来确定合法的更新步长
                for j in range(backtrack_iters):
                    update_kl_divergence, pol_loss_new = set_and_eval_policy_params(step=backtrack_coef ** j)
                    if update_kl_divergence <= self.delta and pol_loss_new <= pol_loss_old:
                        self.logger.to_log('[线搜索]：在第%d次搜索中接受了参数更新' % j)
                        break
                else:
                    # 如果线搜索不成功的话，那么就放弃在此方向上的更新，重新收集数据
                    self.logger.to_log('[线搜索]：线搜索失败！放弃在该方向上更新策略参数，将策略网络参数重置为前置参数')
                    _, _ = set_and_eval_policy_params(step=0)
            # 对估值网络进行训练
            here_feed_dict = {self.obs_ph: b_obs, self.repay_ph: b_repay}
            for _ in range(train_v_iters):
                sess.run(self.train_evaluate, feed_dict=here_feed_dict)
            # 根据更新的估值网络模型计算出估值损失
            here_feed_dict = {self.obs_ph: b_obs, self.repay_ph: b_repay}
            eva_loss_new = sess.run(self.eva_loss, feed_dict=here_feed_dict)
            self.logger.to_log('更新策略网络的广义优势损失为：%.4f; 更新估值网络的估计损失为：%.4f' % (pol_loss_new,
                                                                           eva_loss_new))
        # 开始进行测试和训练
        obs_t, ep_repay, con_index = self.env.reset(), 0, 0
        epoch = 0
        for epoch in range(learn_epochs):
            self.logger.to_log('+++++ 执行 【%d】 次回合' % (epoch+1))
            for t in range(self.max_size_per_epoch):
                # 依据策略网络和估值网络进行决策和估值
                here_feed_dict = {self.obs_ph: obs_t.reshape(1, -1)}
                agent_outs = sess.run([self.pi, self.value, self.logp_pi] + list(self.info_dict.values()),
                                      feed_dict=here_feed_dict)
                # 返回的对象依次为：当前决策给出的行动向量，当前状态观测向量的估值，当前行动向量在决策分布中的对数似然
                # 基于当前状态观测向量映射出的决策分布的参数信息
                act_t, value_t, logp_act_t, info_t = agent_outs[0], agent_outs[1], agent_outs[2], agent_outs[3:]
                # 将决策分布的参数信息封装为字典
                info_dict_t = {}
                for i, k in enumerate(self.info_dict_keys.keys()):
                    info_dict_t[k] = info_t[i]
                # 环境对象执行行动向量，给出奖励信号，衍生状态观测和结束信号
                rew_t, obs_n, done_t, _ = self.env.step(act_t[0])
                # 累加总体奖励和控制次数
                ep_repay += rew_t
                con_index += 1
                # 向缓存器中记录当前回合的状态观测，行动，奖励，估值，对数似然和决策分布参数
                buffer.store(obs_t, act_t, rew_t, value_t, logp_act_t, info_dict_t)
                # 更新状态观测到最新的状态观测
                obs_t = obs_n
                # 判断是否达到了终止状态
                terminal = done_t or (con_index == max_con_len)
                if terminal or (t == self.max_size_per_epoch - 1):
                    if not terminal:
                        self.logger.to_log('[提前中止游戏]：由于训练数据已经写满缓存器，中止游戏交互，开始训练模型')
                    else:
                        if ep_repay > threshold:
                            self.logger.to_log('[控制正常结束]：总得分： %.2f' % ep_repay)
                    last_val = 0 if done_t else sess.run(self.value, feed_dict={self.obs_ph: obs_t.reshape(1, -1)})
                    buffer.finish_path(last_val)
                    obs_t, ep_repay, con_index = self.env.reset(), 0, 0
            # 完成一次对数据缓存器的填充之后，开始对策略网络和估值网络进行更新
            update()
            # 存储模型网络参数
            if epoch % save_freq == 0 and epoch > 5:
                self.logger.to_log('保存网络模型参数')
                model_saver.save(sess, self.target_file_name, global_step=epoch)
        print('[训练终止]：试验结束')
        model_saver.save(sess, self.target_file_name, global_step=epoch)
        sess.close()
        self.env.close()

    def test(self, max_control_len):
        import time
        # 创建计算资源会话
        sess = tf.Session()
        # 创建模型保存对象
        model_saver = tf.train.Saver()
        # 向计算图中写入存储好的模型参数
        model_saver.restore(sess, tf.train.latest_checkpoint(self.target_dir_path))
        # 开始测试模型
        repay = 0
        # 重置环境的状态，并将状态观测修正为标准模式
        obs = self.env.reset()
        obs_t = obs.reshape(1, -1)
        for _ in range(max_control_len):
            try:
                self.env.render()
                time.sleep(0.01)
            except Exception as e:
                pass
            # 基于状态观测进行决策，并由环境对象执行行动
            act_t = sess.run(self.pi, feed_dict={self.obs_ph: obs_t})
            rew_t, obs_n, done_n, _ = self.env.step(act_t[0])
            repay += rew_t
            obs_t = obs_n.reshape(1, -1)

            if done_n:
                print('游戏结束')
                break
        print('本局游戏的总得分：%.2f' % repay)

