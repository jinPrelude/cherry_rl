from learner import run_learner
from agent.dummy import DummyDiscreteAgent
from actor import run_actor

from subprocess import Popen
import signal
import sys


def train(agent, actor_num):
    try:
        commands = []
        for i in range(1, actor_num):
            commands.append(["python", "actor.py", "--id", f"{i}"])
        procs = [Popen(i) for i in commands]
        run_learner(agent=agent)
    except:
        for p in procs:
            print(f"kill {p}")
            p.send_signal(signal.SIGINT)
