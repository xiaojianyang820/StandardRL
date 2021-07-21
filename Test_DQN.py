from ENVS.FlappyBirdEnv import FlappyBirdEnv
from TOOLS.Logger import LoggerPrinter
from DQN.model import DeepQNetwork
from DQN.algo import DQNLearning


def main(game_index, game_mode):
    logger = LoggerPrinter()
    if game_index == 1:
        exp_name = 'FlappyBird'
        gamma = 0.99
        rho = 0.01
        env = FlappyBirdEnv(logger=logger)
        learn_epochs = 55000
        max_iter_per_epoch = 500
        is_retrain_label = False
        evaluate_lr = 1e-6
    dqn_model = DeepQNetwork(env.act_dim)
    dqn = DQNLearning(env=env, evaluate_model=dqn_model, logger=logger, save_dir='MODEL_PARAMS', exp_name=exp_name,
                      rho=rho, gamma=gamma, evaluate_lr=evaluate_lr)
    if game_mode == 'TRAIN':
        dqn.train(learn_epochs=learn_epochs, max_iter_per_epoch=max_iter_per_epoch, is_retrain_label=is_retrain_label,
                  buffer_size=50000, init_epsilon=0.3, final_epsilon=0.0005, eps_de_coef=0.99999)
        dqn.test(test_epochs=20, max_iter_per_epoch=500)
    elif game_mode == 'TEST':
        dqn.test(test_epochs=20, max_iter_per_epoch=500)


if __name__ == '__main__':
    game_index = 1
    game_mode = 'TRAIN'
    main(game_index=1, game_mode='TEST')
