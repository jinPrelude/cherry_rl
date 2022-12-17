from abc import *


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def train(self, minibatch):
        pass
