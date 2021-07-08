import datetime
import abc


class Logger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_log(self, info: str) -> None:
        pass

    @abc.abstractmethod
    def to_warn(self, info: str) -> None:
        pass


class LoggerPrinter(Logger):

    def to_log(self, info: str):
        current_dt = datetime.datetime.now()
        current_dt_str = datetime.datetime.strftime(current_dt, '%Y-%m-%d %H:%M:%S')
        print('[消息] %s : %s' % (current_dt_str, info))

    def to_warn(self, info: str):
        current_dt = datetime.datetime.now()
        current_dt_str = datetime.datetime.strftime(current_dt, '%Y-%m-%d %H:%M:%S')
        print('[警告] %s : %s' % (current_dt_str, info))
