from copy import deepcopy

class ReplayBuffer:
    """ Referenced to minimalRL.
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

    def store(self, actor_id, obs):
        if actor_id not in self.dict.keys():
            self.dict[actor_id] = obs
        else:
            raise AssertionError(f"Try to overwrite existing key: self.dict[{actor_id}][{key}]")

    def delete_by_id(self, actor_id):
        del self.dict[actor_id]

    def is_full(self):
        return True if len(self) >= self.waiting_batch_size else False
    
    def get_all_ids_obs(self):
        id_lst = list(self.dict.keys())
        obs_lst = []
        for actor_id in id_lst: obs_lst.append(self.dict[actor_id])
        return deepcopy(id_lst), deepcopy(obs_lst)

class ProcessedBatch:
    def __init__(self):
        self.lst = {}
    
    def store(self, actor_id, obs, action):
        self.lst[actor_id] = (obs, action)
    
    def get_by_id(self, actor_id):
        return self.lst[actor_id]
    
    def is_id_exist(self, actor_id):
        return True if actor_id in self.lst.keys() else False
    
    def delete_by_id(self, actor_id):
        del self.lst[actor_id]
