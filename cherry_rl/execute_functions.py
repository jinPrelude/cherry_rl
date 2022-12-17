from learner import run_learner
from agent.dummy import DummyDiscreteAgent
from actor import run_actor

from subprocess import Popen
import signal
import sys


def train(agent, actor_args, learner_args):
    try:
        commands = []
        for i in range(1, actor_args["actor_num"] + 1):
            commands.append(
                [
                    "python",
                    "actor.py",
                    "--id",
                    f"{i}",
                    "--env-name",
                    actor_args["env_name"],
                ]
            )
        procs = [Popen(i) for i in commands]
        run_learner(
            agent=agent,
            waiting_batch_size=learner_args["waiting_batch_size"],
            start_training_batch_size=learner_args["start_training_batch_size"],
        )
    except:
        for p in procs:
            print(f"kill {p}")
            p.send_signal(signal.SIGINT)
