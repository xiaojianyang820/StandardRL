import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """
    该类实例化的对象代表了蒙特卡罗搜索树中的一个节点
    """
    def __init__(self, parent_node, selected_prior_prob):
        # 该节点的父节点，也是一个TreeNode对象
        self._parent_node = parent_node
        # 该节点的子节点组，为一个映射，元素是动作->节点，表示在该节点上采取某种动作可以达到的节点
        self._child_nodes = {}
        # 该节点被访问的次数
        self._n_visits = 0
        # 策略估值网络对当前节点状态的评分
        self._Q = 0
        # 对该节点状态评分估计的”标准差“，反映了估计的可靠程度
        self._confidence_STD = 0
        # 由节点状态评分所决定的选择给节点的先验概率
        self._sele_prior_prob = selected_prior_prob

    def select(self, c_puct):
        return max(self._child_nodes.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        self._u = (c_puct * self._sele_prior_prob * np.sqrt(self._parent_node._n_visits) / (1+self._n_visits))
        return self._Q + self._u

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._child_nodes:
                self._child_nodes[action] = TreeNode(self, prob)

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent_node:
            self._parent_node.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return self._child_nodes == {}

    def is_root(self):
        return self._parent_node is None


class MCTS(object):
    def __init__(self, policy_evaluate_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_evaluate_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        # 从搜索树的根节点开始，逐层搜索，
        # 如果该节点是叶节点，那么就停止搜索，开始进行后续工作；
        # 否则的话，就根据蒙特卡罗搜索树选择分支的原则，挑选一个路径，继续向下搜索
        node = self._root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        # 如果搜索到了一个叶节点，就计算该棋面状态的决策分布和估值
        action_probs, leaf_value = self._policy(state)
        # 判断当前的棋面状态是否是终止状态，以及胜者为哪一方
        end, winner = state.game_end()
        # 如果游戏没有结束，那么就在该节点展开其后续可能的行动
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        # 由于在实际训练过程中棋面对应的真实价值是决策执行之后所得到结果
        # 而这里的价值是上一个棋手行动之后得到棋面的估值，如果当前棋手能够在该棋面上得到很高的评分
        # 那么对于上一个棋手而言，就是相反的很低的评分
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temperature=1e-3):
        # 从同一个棋面状态开始，沿着蒙特卡罗搜索树迭代进行前进搜索， 完成整个蒙特卡罗搜索树的构建
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        # 基于根节点去查询各个决策行动的访问次数，生成决策分布
        # 决策分布里面引入了控制随机程度的参数temperature
        act_visits = [(act, node._n_visits) for act, node in self._root._child_nodes.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        # 在自我对弈测试中，由于对战双方都是由同一个决策器给出的，所以给出当前的行动之后，下一个需要解决的棋面就是
        # 当前蒙特卡罗搜索树上根节点对应行动下的子节点所对应的那个棋面
        if last_move in self._root._child_nodes:
            self._root = self._root._child_nodes[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSPlayer(object):
    def __init__(self, policy_evaluate_fn, c_puct=5, n_playout=200, is_selfplay=0):
        self.mcts = MCTS(policy_evaluate_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def get_action(self, board, temp=1e-3, return_prob=0):
        # 查询当前棋盘上的可行状态
        sensible_moves = board.availables
        # 初始化全部的棋盘点位的决策概率为0
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            # 可行的决策点位以及对应的决策概率
            acts, probs = self.mcts.get_move_probs(board, temp)
            # 将全部棋盘点位中那些可行的决策点位填写上合适的决策概率
            move_probs[list(acts)] = probs
            # 如果是自我对弈
            if self._is_selfplay:
                # 就在决策分布上施加一定的随机噪音，进而去产生实际行动
                move = np.random.choice(acts, p=0.75*probs+0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                # 在蒙特卡罗搜索树上推进一步
                self.mcts.update_with_move(move)
            # 如果是真实对战
            else:
                # 那么就按照决策分布去进行行动
                move = np.random.choice(acts, p=probs)
                # 因为真实对战中，决策器走完这一步之后，是其他棋手进行决策，对手的决策是不可知的，所以与当前的蒙特卡罗搜索树
                # 就不一样了，所以需要重置整个蒙特卡罗搜索树
                self.mcts.update_with_move(-1)
                location = board.move_to_location(move)
                print('AI Move: %d; %d\n' % (location[0], location[1]))
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print('[WARNING]: The Board is Full.')

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __str__(self):
        return 'MCTS {}'.format(self.player)
