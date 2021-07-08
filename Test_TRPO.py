from TRPO.algo import TRPO
from TOOLS.Logger import LoggerPrinter
from TRPO.model import MLPContiModel, MLPEvaluateModel, MLPCateModel
from ENVS.Envs import MountainCarContiEnv, PendulumEnv, CartPoleEnv, MountainCarCateEnv, AcrobotEnv
import numpy as np

def main(game_index: int, mode: str) -> None:
    """
    该函数实现了在各个游戏下对TRPO算法进行测试

    :param game_index: int,
        测试游戏环境的编号：
            1. 连续控制下的高山行车
            2. 连续控制下的倒立摆
            3. 离散控制下的平衡摆
            4. 离散控制下的高山行车
            5. 离散控制下的二级摆
    :param mode: str,
        [Train, Test]
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
        max_control_len = 1500
        delta = 0.01
    elif game_index == 2:
        exp_name = 'Pendulum'
        env = PendulumEnv(logger=logger)
        act_lim = np.array([2., ])
        gamma = 0.95
        beta = 0.92
        learn_epoch = 100
        max_control_len = 1200
        # 在策略改进的早期可以允许kl散度比较大，而随着策略的优化，KL散度应该越来越小，以保证
        # 在策略达成较优状态之后再出现震荡现象
        delta = 0.005
    elif game_index == 3:
        exp_name = 'CartPole'
        env = CartPoleEnv(logger=logger)
        gamma = 0.95
        beta = 0.9
        learn_epoch = 60
        max_control_len = 500
        # 在策略改进的早期可以允许kl散度比较大，而随着策略的优化，KL散度应该越来越小，以保证
        # 在策略达成较优状态之后再出现震荡现象
        delta = 0.005
    elif game_index == 4:
        exp_name = 'MountainCarCate'
        env = MountainCarCateEnv(logger=logger)
        gamma = 0.99
        beta = 0.99
        learn_epoch = 120
        max_control_len = 2000
        delta = 0.01
    elif game_index == 5:
        exp_name = 'Acrobot'
        env = AcrobotEnv(logger=logger)
        gamma = 0.95
        beta = 0.92
        learn_epoch = 80
        max_control_len = 1500
        delta = 0.005

    if game_index in [1, 2]:
        policy_model = MLPContiModel(env.obs_dim, env.act_dim, hidden_size=(64, 64), hd_activation='Tanh',
                                     act_lim=act_lim, logger=logger)
        info_dict_keys = {'Mean': env.act_dim, 'Log_STD': env.act_dim}
    else:
        policy_model = MLPCateModel(env.obs_dim, env.act_dim, hidden_size=(64, 64), hd_activation='Sigmoid',
                                    logger=logger)
        info_dict_keys = {'logp_vec': env.act_dim}
    evaluate_model = MLPEvaluateModel(env.obs_dim, hidden_size=(64, 64), hd_activation='Tanh', logger=logger)
    trpo = TRPO(env=env, policy_model=policy_model, evaluate_model=evaluate_model, model_save_path='MODEL_PARAMS',
                exp_name=exp_name, logger=logger, gamma=gamma, beta=beta, damping_coef=0.01,
                info_dict_keys=info_dict_keys, max_size_per_epoch=5000, delta=delta,
                eva_lr=0.001, algo='trpo')
    if mode == 'Train':
        trpo.train(retrain_label=False, cg_iters=10, backtrack_iters=10, backtrack_coef=0.9, train_v_iters=100,
                   learn_epochs=learn_epoch, threshold=-20000, max_con_len=max_control_len)
        trpo.test(max_control_len=max_control_len)
    else:
        trpo.test(max_control_len=max_control_len)


if __name__ == '__main__':
    main(5, 'Train')

