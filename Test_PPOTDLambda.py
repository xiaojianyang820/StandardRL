from ENVS.Envs import PendulumEnv
from PPO_TD_Lambda.model import MLPContiControlModel, MLPEvaluateModel
from PPO_TD_Lambda.algo_2 import PPOTDLambda
from PPO_TD_Lambda.algo import PPOTDLamda as PPO1
from TOOLS.Logger import LoggerPrinter
import numpy as np
"""
本测试完成了对PPO_TD_Lambda算法在倒立摆上的运行效果，
game_index=1: 在algo实现上测试
game_index=2: 在algo_2实现上测试
PPO算法在倒立摆问题上控制效果都差一些，只有对奖励函数做一些修正，即(rew+8)/8之后才能获得较好的效果。
这说明了奖励信号如果是连续的，那么最后通过一定的数值计算来修正，确保奖励信号的分布和估值网络的初始分布向接近。
"""


def main(game_index, game_mode):
    logger = LoggerPrinter()
    if game_index == 1:
        exp_name = 'Pendulum'
        env = PendulumEnv(logger=logger)
        act_lim = np.array([2., ])
        gamma = 0.90
        learn_epoch = 900
        max_control_len = 200
        clip_ratio = 0.2
    elif game_index == 2:
        exp_name = 'Pendulum'
        env = PendulumEnv(logger=logger)
        act_lim = np.array([2., ])
        gamma = 0.90
        learn_epoch = 900
        max_control_len = 200
        clip_ratio = 0.2

    if game_index in [1, 2]:
        policy_model = MLPContiControlModel(env.obs_dim, env.act_dim, hidden_size=(100,), hd_activation='ReLU',
                                            max_control_lim=act_lim, logger=logger)
    else:
        policy_model = None
    evaluate_model = MLPEvaluateModel(env.obs_dim, hidden_size=(100,), hd_activation='ReLU', logger=logger)

    if game_mode == 'TRAIN':
        if game_index == 1:
            ppotdlambda = PPOTDLambda(env=env, policy_model=policy_model, evaluate_model=evaluate_model,
                                      model_dir='MODEL_PARAMS', exp_name=exp_name, logger=logger, gamma=gamma,
                                      clip_ratio=clip_ratio, policy_lr=0.0001, evaluate_lr=0.0002,
                                      conti_act_lim=act_lim)
            ppotdlambda.train(retrain_label=False, max_iter_per_epoch=max_control_len, learning_epoch=learn_epoch,
                              mini_batch=32, update_num=10, save_freq=100)
        elif game_index == 2:
            ppotdlambda = PPO1(env=env, policy_model=policy_model, evaluate_model=evaluate_model,
                               model_save_dir='MODEL_PARAMS', exp_name=exp_name, logger=logger, gamma=gamma,
                               clip_ratio=clip_ratio, policy_lr=0.0001, evaluate_lr=0.0001, is_ou_noise=False,
                               act_lim=act_lim)
            ppotdlambda.train(1000, 200, 32, False, 10, 100)


if __name__ == '__main__':
    game_index = 2
    game_mode = 'TRAIN'
    main(game_index, game_mode)
