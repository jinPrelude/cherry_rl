from str_arr_converters import vector_gym_arr2str

import random
import argparse

import grpc
import cherry_rl_pb2
import cherry_rl_pb2_grpc

import gym
import numpy as np

def actor_loop(stub, env, actor_id):
    print(f"actor_{actor_id} started.")
    while True:
        obs, _ = env.reset()
        obs = vector_gym_arr2str(obs)
        done, reward = 'False', '0'
        response = stub.DiscreteGymStep(cherry_rl_pb2.CallRequest(actor_id=actor_id, obs=obs, reward=reward, done=done, request_type="reset"), wait_for_ready=True)
        while True:
            action = int(response.action)
            obs, reward, done, _, _ = env.step(action)
            obs, reward, done = vector_gym_arr2str(obs), str(reward), str(done)
            response = stub.DiscreteGymStep(cherry_rl_pb2.CallRequest(actor_id=actor_id, obs=obs, reward=reward, done=done, request_type="step"))
            if done:
                break


def run_actor(actor_id: int):
    assert actor_id >= 1, "actor_id should started from 1."

    channel = grpc.insecure_channel('localhost:50051')
    stub = cherry_rl_pb2_grpc.CherryRLStub(channel)

    env = gym.make("CartPole-v1")

    actor_loop(stub, env, actor_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int)
    args = parser.parse_args()
    run_actor(args.id)