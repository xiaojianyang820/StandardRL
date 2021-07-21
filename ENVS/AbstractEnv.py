import abc
from abc import ABC

import numpy as np
from TOOLS.Logger import Logger


class AbsEnv(metaclass=abc.ABCMeta):
    """
    该类是一个抽象类，规定了一个标准环境的API接口，任何具体环境类都需要继承自该抽象类，
    各个具体环境类不得添加新的对外接口
    """
    def __init__(self, logger: Logger = None):
        """
        在环境初始化的过程中，必须包含如下四个属性：
            1. 状态观测的维度obs_dim(int)
            2. 行动向量的维度act_dim(int)
            3. 状态观测的类型obs_type(str,[Categorical, Continuous])
            4. 行动向量的类型act_type(str,[Categorical, Continuous])

        :param logger: Logger对象
            用于记录训练日志的日志对象
        :return
        """
        self.obs_dim = self._obs_dim()
        self.act_dim = self._act_dim()
        self.obs_type = self._obs_type()
        self.act_type = self._act_type()

        self.logger = logger

    def check_obs(self, obs: object) -> bool:
        """
        检验状态观测向量obs是否合法

        :param obs: object,
            待校验是否合法的状态观测向量
        :return: bool,
            如果状态观测向量是合法的，那么就返回True
            如果状态观测向量是非法的，那么就返回False
        """
        try:
            assert isinstance(obs, np.ndarray)
        except AssertionError as e:
            self.logger.to_warn("环境对象返回的状态观测类型不正确，必须是np.ndarray")
            return False

        try:
            assert len(obs.shape) == 1
        except AssertionError as e:
            self.logger.to_warn('环境对象返回的状态观测向量形状不正确，只能够是一维向量')
            return False

        return True

    def check_act(self, act: object) -> bool:
        """
        检验行动向量act是否合法

        :param act: object,
            待校验是否合法的行动向量
        :return: bool,
            如果行动向量是合法的，那么就返回True
            如果行动向量是非法的，那么就返回False
        """
        try:
            assert isinstance(act, np.ndarray)
        except AssertionError as e:
            self.logger.to_warn('决策器返回的行动向量类型不正确，必须是np.ndarray')
            return False
        try:
            assert len(act.shape) == 1
        except AssertionError as e:
            self.logger.to_warn('决策器返回的行动向量形状不正确，只能够是一维向量')
            return False

        return True

    def check_reward(self, reward: object) -> bool:
        """
        检验奖励信号是否合法

        :param reward: object,
            待检验是否合法的奖励信号
        :return: bool,
            如果奖励信号是合法的，那么就返回True
            如果奖励信号是非法的，那么就返回False
        """
        try:
            assert isinstance(float(reward), float)
        except Exception as e:
            self.logger.to_warn('环境对象返回的奖励信号类型不正确，只能是float类型')
            return False

        return True

    def reset(self) -> np.ndarray:
        """
        重置环境至初始状态
        继承该环境抽象类的子类必须实现一个函数_reset，以供该函数调用，
        _reset函数的返回值必须是np.array，形状为1维向量

        :return: np.ndarray
            初始的状态观测向量
        """
        obs = self._reset()
        check_label = self.check_obs(obs)
        if check_label:
            return obs
        else:
            self.logger.to_warn('错误的状态观测向量为：' + str(obs))
            raise Exception('环境对象的初始化函数返回的状态观测向量不符合要求')

    def step(self, act: np.ndarray) -> tuple:
        """
        环境对象执行决策器产生的行动向量

        :param act: np.ndarray,
            待执行的行动向量
        :return: tuple,
            返回由(reward(float), next_obs(1-d array), done(bool), info(str))构成的tuple
        """
        check_label = self.check_act(act)
        if not check_label:
            self.logger.to_warn('错误的行动向量为：' + str(act))
            raise Exception('决策器产生的行动向量不符合要求')
        next_obs, reward, done, info = self._step(act)
        check_label = self.check_reward(reward)
        if not check_label:
            self.logger.to_warn('错误的奖励信号为：' + str(check_label))
            raise Exception('环境对象返回的奖励信号不符合要求')
        check_label = self.check_obs(next_obs)
        if not check_label:
            self.logger.to_warn('错误的衍生状态观测向量：' + str(next_obs))
            raise Exception('环境对象返回的衍生状态观测信号不符合要求')
        try:
            assert isinstance(done, bool)
        except AssertionError as e:
            self.logger.to_warn('错误的回合结束信号：' + str(done))
            raise e
        return reward, next_obs, done, info

    def render(self) -> None:
        """
        部分具有可视化界面的环境对象刷新可视化前端

        :return: None,
        """
        self._render()

    def close(self) -> None:
        """
        关闭整个动态环境

        :return: None,
        """
        self._close()

    @abc.abstractmethod
    def _obs_dim(self):
        pass

    @abc.abstractmethod
    def _act_dim(self):
        pass

    @abc.abstractmethod
    def _act_type(self):
        pass

    @abc.abstractmethod
    def _obs_type(self):
        pass

    @abc.abstractmethod
    def _reset(self):
        pass

    @abc.abstractmethod
    def _step(self, act):
        pass

    @abc.abstractmethod
    def _render(self):
        pass

    @abc.abstractmethod
    def _close(self):
        pass


class AbsImageEnv(AbsEnv, ABC):
    """
        该类是用于处理基于图像的增强学习问题的一个抽象类，规定了一个标准环境的API接口，任何具体环境类都需要继承自该抽象类，
        各个具体环境类不得添加新的对外接口
    """
    def __init__(self, logger: Logger = None):
        """
        在环境初始化的过程中，必须包含如下四个属性：
            1. 状态观测的维度obs_dim(tuple[int])
            2. 行动向量的维度act_dim(int)
            3. 状态观测的类型obs_type(str,[Categorical, Continuous])
            4. 行动向量的类型act_type(str,[Categorical, Continuous])

        :param logger: Logger对象
            用于记录训练日志的日志对象
        :return
        """
        super(AbsImageEnv, self).__init__(logger=logger)
        # 需要确保该类对象中的obs_dim是描述图像长宽高的一个tuple
        assert isinstance(self.obs_dim, tuple)

    def check_obs(self, obs: object) -> bool:
        """
        检验状态观测向量obs是否合法

        :param obs: object,
            待校验是否合法的状态观测向量
        :return: bool,
            如果状态观测向量是合法的，那么就返回True
            如果状态观测向量是非法的，那么就返回False
        """
        try:
            assert isinstance(obs, np.ndarray)
        except AssertionError as e:
            self.logger.to_warn("环境对象返回的状态观测类型不正确，必须是np.ndarray")
            return False

        return True

