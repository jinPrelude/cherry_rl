from concurrent import futures
import logging
import math
import time
import random
from copy import deepcopy
import grpc
import seed_rl_pb2
import seed_rl_pb2_grpc

import numpy as np
class ReplayBuffer:
    """ Referenced by minimalRL 
    """
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def store(self, transition):
        print(f"replay len: {len(self)}")
        self.data.append(transition)
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, 

class WaitingBatch:
    def __init__(self, waiting_batch_size):
        self.waiting_batch_size = waiting_batch_size
        self.dict = {}

    def __len__(self):
        return len(self.dict)

    def store_by_id(self, actor_id, key: str, value: object):
        assert key in ["obs", "reward", "action", "next_obs", "done"]
        if actor_id not in self.dict.keys() : self.dict[actor_id] = {}
        if key not in self.dict[actor_id].keys():
            self.dict[actor_id][key] = value
        else:
            raise AssertionError(f"Try to overwrite existing key: self.dict[{actor_id}][{key}]")

    def get_by_id(self, actor_id, key):
        # print(f"{actor_id} : {self.dict[actor_id]}")
        return self.dict[actor_id][key]
    
    def check_key_exist_by_id(self, actor_id, key):
        return True if key in self.dict[actor_id].keys() else False
    
    def delete_by_id(self, actor_id):
        del self.dict[actor_id]

    def is_full(self):
        return True if len(self) >= self.waiting_batch_size else False
    
    def get_all_ids_obs(self):
        id_lst = list(self.dict.keys())
        obs_lst = []
        for actor_id in id_lst: obs_lst.append(self.dict[actor_id]["obs"])
        return deepcopy(id_lst), deepcopy(obs_lst)
class ProcessedBatch:
    def __init__(self):
        self.lst = {}
    
    def store(self, actor_id, obs, action):
        self.lst[actor_id] = (obs, action)
    
    def get_by_id(self, actor_id):
        result = self.lst[actor_id]
        return result
    
    def keys(self):
        return self.lst.keys()
    
    def is_id_exist(self, actor_id):
        return True if actor_id in self.lst.keys() else False
    
    def delete_by_id(self, actor_id):
        del self.lst[actor_id]
    
class SeedRLServicer(seed_rl_pb2_grpc.SeedRLServicer):

    def __init__(self, waiting_batch_size = 4):
        self.Agent = random.randint
        self.actor_num = 2
        self.waiting_batch = WaitingBatch(waiting_batch_size = waiting_batch_size)
        self.processed_batch = ProcessedBatch()
        self.replay_buffer= ReplayBuffer()
        self.actors_ready = np.zeros(self.actor_num)

    
    def _check_actors_ready(self):
        return np.all(self.actors_ready == 1)

    def _process(self):
        id_lst, obs_lst = self.waiting_batch.get_all_ids_obs()
        for actor_id, obs in zip(id_lst, obs_lst):
            action = self.Agent(0, 1)
            self.processed_batch.store(actor_id, obs, action)
            self.waiting_batch.delete_by_id(actor_id)

    def _is_processed(self, actor_id):
        return True if actor_id in self.processed_batch.keys() else False

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
            return seed_rl_pb2.DiscreteGymReply(action = -1)
        else:
            self.waiting_batch.store_by_id(request.actor_id, "obs", obs)

            # TODO : actor를 여려개 만드니 문제가 생긴다. 여기서 batch_is_full 이랑 self._process()에 접근하면서 충돌하는듯.
            while not self.processed_batch.is_id_exist(request.actor_id):
                if self.waiting_batch.is_full():
                    self._process()
                time.sleep(0.001)
            _, action = self.processed_batch.get_by_id(request.actor_id)
            return seed_rl_pb2.DiscreteGymReply(action = self.Agent(0, 1)+1)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    seed_rl_pb2_grpc.add_SeedRLServicer_to_server(
        SeedRLServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()