from batch_utils import WaitingBatch, ProcessedBatch, ReplayBuffer
from agent.abstracts import Agent
from agent.dummy import DummyDiscreteAgent

from concurrent import futures
import time
import threading
import random
from copy import deepcopy

import grpc
import cherry_rl_pb2
import cherry_rl_pb2_grpc

import numpy as np

class CherryRLServicer(cherry_rl_pb2_grpc.CherryRLServicer):

    def __init__(self, agent: Agent, waiting_batch_size: int = 3):
        self.Agent = agent
        self.waiting_batch = WaitingBatch(waiting_batch_size = waiting_batch_size)
        self.processed_batch = ProcessedBatch()
        self.replay_buffer = ReplayBuffer()

        batching_thread = threading.Thread(target=self.batching_thread)
        batching_thread.daemon = True
        batching_thread.start()

    def batching_thread(self):
        while True:
            if self.waiting_batch.is_full():
                id_lst, obs_lst = self.waiting_batch.get_all_ids_obs()
                for actor_id, obs in zip(id_lst, obs_lst):
                    action = self.Agent.forward(obs)
                    self.processed_batch.store(actor_id, obs, action)
                    self.waiting_batch.delete_by_id(actor_id)
            else:
                time.sleep(0.0000000001)

    def DiscreteGymStep(self, request, context):
        done = False
        if request.request_type == "reset":
            obs = request.obs
        elif request.request_type == "step":
            # get transitions from the request
            next_obs = request.obs
            reward = request.reward
            done = True if request.done == "True" else False
            # move transition to replay buffer
            obs, action = self.processed_batch.get_by_id(request.actor_id)
            self.replay_buffer.store((obs, action, reward, next_obs, done))
            self.processed_batch.delete_by_id(request.actor_id)
            # obs to next_obs
            obs = next_obs
        # store obs and actor_id in the waiting_batch
        if done:
            return cherry_rl_pb2.DiscreteGymReply(action = -1)
        else:
            self.waiting_batch.store(request.actor_id, obs)
            while not self.processed_batch.is_id_exist(request.actor_id):
                time.sleep(0.0000000001)
            _, action = self.processed_batch.get_by_id(request.actor_id)
            return cherry_rl_pb2.DiscreteGymReply(action = self.Agent.forward(obs)+1)

def run_learner(agent: Agent):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cherry_rl_pb2_grpc.add_CherryRLServicer_to_server(
        CherryRLServicer(agent=agent), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    agent = DummyDiscreteAgent(0, 1)
    run_learner(agent)