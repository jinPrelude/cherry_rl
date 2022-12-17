from execute_functions import train
from agent.dummy import DummyDiscreteAgent


if __name__ == "__main__":
    agent = DummyDiscreteAgent(0, 1)
    train(agent, 10)
