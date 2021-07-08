from SAC.algo import SAC
from SAC.model import MLPContiModel, MLPEvaluateModel
from TOOLS.Logger import LoggerPrinter
from ENVS.Envs import MountainCarContiEnv, PendulumEnv
import numpy as np


def main(game_index: int, mode: str) -> None:
    """
    该函数实现了在各个游戏下对DDPG算法进行测试
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
        policy_model = MLPContiModel(env.obs_dim, env.act_dim, (30, 15), 'Sigmoid', logger)
        evaluate_model = MLPEvaluateModel(env.obs_dim, env.act_dim, (30, 15), 'Sigmoid', logger)
        gamma = 0.95
        eva_lr = 0.005
        pol_lr = 0.005
        rho = 0.005
        learn_epochs = 25
        max_iter_per_epoch = 1500
        alpha = 0.3
        is_OU_noise = False

    elif game_index == 2:
        exp_name = 'MountainCarConti'
        env = MountainCarContiEnv(logger=logger)
        act_high = np.array([1., ])
        policy_model = MLPContiModel(env.obs_dim, env.act_dim, (40, 25), 'Sigmoid', logger)
        evaluate_model = MLPEvaluateModel(env.obs_dim, env.act_dim, (30, 15), 'Sigmoid', logger)
        gamma = 0.99
        eva_lr = 0.005
        pol_lr = 0.005
        rho = 0.005
        learn_epochs = 100
        max_iter_per_epoch = 1500
        alpha = 0.3
        is_OU_noise = True

    sac = SAC(env, policy_model, evaluate_model, logger, 'MODEL_PARAMS', exp_name, gamma=gamma,
              eva_lr=eva_lr, pol_lr=pol_lr, rho=rho, conti_control_max=act_high, alpha=alpha,
              is_OU_noise=is_OU_noise)
    if mode == 'TRAIN':
        sac.train(learn_epochs=learn_epochs, max_iter_per_epoch=max_iter_per_epoch, retrain_label=False,
                  buffer_size=100000, start_control_step=3000, update_freq=100, sample_size=100, save_freq=1000)
        sac.test(test_epochs=10, max_iter_per_epoch=1200)
    elif mode == 'TEST':
        sac.test(test_epochs=10, max_iter_per_epoch=1500)


if __name__ == '__main__':
    main(2, 'TRAIN')
