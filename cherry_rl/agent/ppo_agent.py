from agent.abstracts import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PPOModel(nn.Module):
    def __init__(self, state_num, action_num, learning_rate=0.0005):
        super(PPOModel, self).__init__()
        self.state_num = state_num
        self.action_num = action_num

        self.fc1   = nn.Linear(state_num,256)
        self.fc_pi = nn.Linear(256,action_num)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

class PPOAgent(Agent):
    def __init__(self, state_num, action_num):
        super(PPOAgent, self).__init__()
        self.state_num = state_num
        self.action_num = action_num
    
        self.model = PPOModel(self.state_num, self.action_num)

    def forward(self, obs):
        obs = torch.from_numpy(obs).float()
        prob = self.model.pi(obs)
        m = Categorical(prob)
        action = m.sample().item()
        return action

    def _make_batch(self, minibatch):
            s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
            for transition in minibatch:
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
            return s, a, r, s_prime, done_mask, prob_a

    def train(self, minibatch):
        s, a, r, s_prime, done_mask, prob_a = self._make_batch(minibatch)

        for i in range(K_epoch):
            td_target = r + gamma * self.model.v(s_prime) * done_mask
            delta = td_target - self.model.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.model.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(s) , td_target.detach())

            self.model.optimizer.zero_grad()
            loss.mean().backward()
            self.model.optimizer.step()