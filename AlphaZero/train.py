import random
import numpy as np
from collections import deque
from game import Board, Game
from mcts import MCTSPlayer
from policy_evaluate_model import PolicyEvaluateNet


class TrainPipeLine(object):
    def __init__(self, init_model=None):
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.temp = 1.0
        self.c_puct = 5
        self.n_playout = 400
        self.learn_rate = 5e-3
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.check_freq = 50
        self.game_batch_num = 3000
        if init_model:
            self.policy_evaluate_net = PolicyEvaluateNet(self.board_width, self.board_height, model_file=init_model)
        else:
            self.policy_evaluate_net = PolicyEvaluateNet(self.board_width, self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_evaluate_net.policy_evaluate_fn, c_puct=self.c_puct,
                                      n_playout=self.n_playout, is_selfplay=1)

    def run(self):
        try:
            for i in range(self.game_batch_num):
                episode_len = self.collect_selfplay_data()
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print(("batch i:{}, "
                           "episode_len:{}, "
                           "loss:{:.4f}, "
                           "entropy:{:.4f}"
                           ).format(i + 1,
                                    episode_len,
                                    loss,
                                    entropy))
                else:
                    print("batch i:{}, "
                          "episode_len:{}".format(i + 1, episode_len))
                # 定期保存模型
                if (i + 1) % self.check_freq == 0:
                    self.policy_evaluate_net.save_model(
                        './current_policy.model')
        except KeyboardInterrupt:
            print('\n quit')

    def collect_selfplay_data(self):
        winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
        play_data = list(play_data)[:]
        episode_len = len(play_data)
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)
        return episode_len

    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)
                ), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        loss, entropy = self.policy_evaluate_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                            self.learn_rate)
        return loss, entropy


if __name__ == '__main__':
    training_pipeline = TrainPipeLine()
    training_pipeline.run()
