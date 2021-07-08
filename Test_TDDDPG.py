from TwinDelayedDDPG.algo import TwinDelayedDDPG
from TwinDelayedDDPG.model import MLPContiModel, MLPEvaluateModel
from TOOLS.Logger import LoggerPrinter
from ENVS.Envs import MountainCarContiEnv, PendulumEnv
import numpy as np


def main(game_index: int, mode: str) -> None:
    """
    该函数实现了在各个游戏下对TDDDPG算法进行测试
    :param game_index: int,
        测试游戏环境的编号：
            1. 连续控制下的倒立摆
            2. 连续控制下的高山行车
    :param mode: str,
        控制模式，[TRAIN, TEST]
    :return: None,
    """
    logger = LoggerPrinter()
    if game_index == 1:
        exp_name = 'Pendulum'
        env = PendulumEnv(logger=logger)
        act_high = np.array([2., ])
        act_low = np.array([-2., ])
        policy_model = MLPContiModel(env.obs_dim, env.act_dim, (30, 15), 'Sigmoid', logger)
        evaluate_model = MLPEvaluateModel(env.obs_dim, env.act_dim, (30, 15), 'Sigmoid', logger)
        gamma = 0.95
        eva_lr = 0.005
        pol_lr = 0.005
        rho = 0.005
        learn_epochs = 50
        max_iter_per_epoch = 1500
        is_OU_noise = False

    elif game_index == 2:
        exp_name = 'MountainCarConti'
        env = MountainCarContiEnv(logger=logger)
        act_high = np.array([1., ])
        act_low = np.array([-1., ])
        policy_model = MLPContiModel(env.obs_dim, env.act_dim, (40, 25), 'Sigmoid', logger)
        evaluate_model = MLPEvaluateModel(env.obs_dim, env.act_dim, (30, 15), 'Sigmoid', logger)
        gamma = 0.99
        eva_lr = 0.005
        pol_lr = 0.005
        rho = 0.005
        learn_epochs = 100
        max_iter_per_epoch = 1500
        is_OU_noise = True

    tdddpg = TwinDelayedDDPG(env, policy_model, evaluate_model, 'MODEL_PARAMS', exp_name, logger, gamma=gamma,
                eva_lr=eva_lr, pol_lr=pol_lr, rho=rho, conti_act_high=act_high, conti_act_low=act_low,
                is_OU_noise=is_OU_noise)
    if mode == 'TRAIN':
        tdddpg.train(buffer_size=1000000, retrain_label=False, learn_epochs=learn_epochs,
                   max_iter_per_epoch=max_iter_per_epoch, sample_size=200, save_freq=100, noise_scale=0.1,
                   start_steps=20000, update_after=10000, update_every=50)
        tdddpg.test(test_epochs=10, max_iter_per_epoch=2000)
    elif mode == 'TEST':
        tdddpg.test(test_epochs=10, max_iter_per_epoch=2000)


if __name__ == '__main__':
    main(2, 'TRAIN')


