from agent.abstracts import Agent

import random

class DummyDiscreteAgent(Agent):
    def __init__(self, action_min, action_max):
        self.action_max = action_max
        self.action_min = action_min
    def forward(self, obs):
        result = random.randint(self.action_min, self.action_max)
        return result
    def train(self):
        pass
