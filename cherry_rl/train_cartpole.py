from execute_functions import train
from agent.ppo_agent import PPOAgent

if __name__ == "__main__":
    agent = PPOAgent(4, 2)
    actor_args = {
        "env_name": "CartPole-v1",
        "actor_num": 2,
    }
    learner_args = {
        "waiting_batch_size": actor_args["actor_num"],
        "start_training_batch_size": 20 * actor_args["actor_num"],
    }
    train(agent, actor_args, learner_args)