from PPO.algo import PPO
from TOOLS.Logger import LoggerPrinter
from PPO.model import MLPContiModel, MLPEvaluateModel, MLPCateModel
from ENVS.Envs import MountainCarContiEnv, PendulumEnv, CartPoleEnv, MountainCarCateEnv, AcrobotEnv
import numpy as np


def main(game_index: int, mode: str) -> None:
    """
    该函数实现了在各个游戏下对PPO算法进行测试

    :param game_index: int,
        测试游戏环境的编号：
            1. 连续控制下的高山行车
            2. 连续控制下的倒立摆
            3. 离散控制下的平衡摆
            4. 离散控制下的高山行车（目前控制效果不好）
            5. 离散控制下的二级摆
    :param mode: str,
        [TRAIN, TEST]
    :return: None,
    """
    logger = LoggerPrinter()
    if game_index == 1:
        exp_name = 'MountainCarConti'
        env = MountainCarContiEnv(logger=logger)
        act_lim = np.array([1., ])
        gamma = 0.999
        beta = 0.95
        learn_epoch = 50
        max_control_len = 5500
        clip_ratio = 0.1
        is_ou_noise = True
    elif game_index == 2:
        exp_name = 'Pendulum'
        env = PendulumEnv(logger=logger)
        act_lim = np.array([2., ])
        gamma = 0.98
        beta = 0.96
        learn_epoch = 600
        max_control_len = 1200
        # 在策略改进的早期可以允许kl散度比较大，而随着策略的优化，KL散度应该越来越小，以保证
        # 在策略达成较优状态之后再出现震荡现象
        clip_ratio = 0.05
        is_ou_noise = False
    elif game_index == 3:
        exp_name = 'CartPole'
        env = CartPoleEnv(logger=logger)
        gamma = 0.95
        beta = 0.9
        learn_epoch = 60
        max_control_len = 500
        # 在策略改进的早期可以允许kl散度比较大，而随着策略的优化，KL散度应该越来越小，以保证
        # 在策略达成较优状态之后再出现震荡现象
        clip_ratio = 0.05
        is_ou_noise = False
    elif game_index == 4:
        exp_name = 'MountainCarCate'
        env = MountainCarCateEnv(logger=logger)
        gamma = 0.99
        beta = 0.99
        learn_epoch = 120
        max_control_len = 2000
        clip_ratio = 0.05
        is_ou_noise = False
    elif game_index == 5:
        exp_name = 'Acrobot'
        env = AcrobotEnv(logger=logger)
        gamma = 0.95
        beta = 0.92
        learn_epoch = 80
        max_control_len = 1500
        clip_ratio = 0.05
        is_ou_noise = False

    if game_index in [1, 2]:
        policy_model = MLPContiModel(env.obs_dim, env.act_dim, hidden_size=(15, 15), hd_activation='Tanh',
                                     conti_control_max=act_lim, logger=logger)
    else:
        policy_model = MLPCateModel(env.obs_dim, env.act_dim, hidden_size=(64, 64), hd_activation='Sigmoid',
                                    logger=logger, conti_control_max=None)
    evaluate_model = MLPEvaluateModel(env.obs_dim, hidden_size=(32, 16), hd_activation='Tanh', logger=logger)
    ppo = PPO(env=env, policy_model=policy_model, evaluate_model=evaluate_model, model_save_dir='MODEL_PARAMS',
              exp_name=exp_name, gamma=gamma, beta=beta, logger=logger, clip_ratio=clip_ratio, policy_lr=0.02,
              evaluate_lr=0.05, is_ou_noise=is_ou_noise)
    if mode == 'TRAIN':
        ppo.train(learning_epochs=learn_epoch, retrain_label=False, save_freq=10,
                  max_iter_per_epoch=max_control_len)
        ppo.test(5, max_control_len)
    else:
        ppo.test(5, max_control_len)


if __name__ == '__main__':
    main(2, 'TRAIN')

