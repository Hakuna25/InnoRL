import numpy as np
from torch.autograd import Variable
import torch
import ray


class MemoryStore:
    def __init__(self, capacity, size):
        self.memory = np.zeros((capacity, size))
        self.memory_counter = 0
        self.capacity = capacity
        self.size = size

    def push(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r, done], s_))  # horizontally stack these vectors
        index = self.memory_counter % self.capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def sample(self, batch_size, state_dim, action_dim):
        sample_index = np.random.choice(self.capacity, batch_size)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :state_dim]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, state_dim:state_dim + action_dim].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, state_dim + action_dim:state_dim + 2]))
        b_done_ = Variable(torch.FloatTensor(b_memory[:, state_dim + 2:state_dim + 3]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -state_dim:]))

        return b_s, b_a, b_r, b_s_, b_done_

    def sampleppo(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for n in self.memory_counter:
            s, a, r, s_, done = self.memory[n]
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([[a]], dtype=torch.float))
            l_r.append(torch.tensor([[r]], dtype=torch.float))
            l_s_.append(torch.tensor([s_], dtype=torch.float))
            l_done.append(torch.tensor([[done]], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        a = torch.cat(l_a, dim=0)
        r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        done = torch.cat(l_done, dim=0)
        return s, a, r, s_, done

    def put(self):
        return ray.put(self.memory)


class OffPolicyBaseReplayBuffer:
    def __init__(self, config, log_prob_dim=1, reward_dim=1, discrete=True):
        capacity = config.buffer_size
        self.capacity = capacity
        self.memory_counter = 0
        self.state = np.zeros((capacity, config.state_space))
        self.next_state = np.zeros((capacity, config.state_space))
        if discrete:
            self.action = np.zeros((capacity, 1))
        else:
            self.action = np.zeros((capacity, config.action_space))
        self.log_prob = np.zeros((capacity, log_prob_dim))
        self.reward = np.zeros((capacity, reward_dim))
        self.done = np.zeros((capacity, 1))

    def store(self, s, a, r, s_, log_prob=None, done=None):
        index = self.memory_counter % self.capacity
        self.state[index] = s
        self.action[index] = a
        self.reward[index] = r
        self.next_state[index] = s_

        if log_prob is not None:
            self.log_prob[index] = log_prob

        if done is not None:
            self.done[index] = done

        self.memory_counter += 1

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.capacity, self.memory_counter), batch_size)
        return {
            'state': torch.tensor(self.state[sample_index]),
            'action': torch.tensor(self.action[sample_index]),
            'reward': torch.tensor(self.reward[sample_index]),
            'next_state': torch.tensor(self.next_state[sample_index]),
            'done': torch.tensor(self.done[sample_index]),
            'log_prob': torch.tensor(self.log_prob[sample_index])
        }


class OffPolicyReplayBuffer:
    def __init__(self, config, log_prob_dim=1, reward_dim=1, discrete=True):
        capacity = config.buffer_size
        self.capacity = capacity
        self.memory_counter = 0
        self.n_agnets = 1
        self.state = np.zeros((capacity, config.n_rollout_threads, config.n_agents, config.state_space))
        self.next_state = np.zeros((capacity, config.n_rollout_threads, config.n_agents, config.state_space))
        if discrete:
            self.action = np.zeros((capacity, config.n_rollout_threads, config.n_agents, 1))
        else:
            self.action = np.zeros((capacity, config.n_rollout_threads, config.n_agents, config.action_space))
        self.log_prob = np.zeros((capacity, config.n_rollout_threads, config.n_agents, log_prob_dim))
        self.reward = np.zeros((capacity, config.n_rollout_threads, reward_dim))
        self.done = np.zeros((capacity, config.n_rollout_threads, 1))

    def store(self, s, a, r, s_, log_prob=None, done=None):
        index = self.memory_counter % self.capacity
        self.state[index] = s
        self.action[index] = a
        self.next_state[index] = s_
        if log_prob is not None:
            self.log_prob[index] = log_prob
        if done is not None:
            self.done[index] = np.reshape(done, (done.shape[0], 1))
        self.reward[index] = np.reshape(r, (r.shape[0], 1))
        self.memory_counter += 1

    def sample(self, batch_size):
        sample_index = np.random.choice(min(self.capacity, self.memory_counter), batch_size)
        return {
            'state': torch.tensor(self.state[sample_index]),
            'action': torch.tensor(self.action[sample_index]),
            'reward': torch.tensor(self.reward[sample_index]),
            'next_state': torch.tensor(self.next_state[sample_index]),
            'done': torch.tensor(self.done[sample_index]),
            'log_prob': torch.tensor(self.log_prob[sample_index])
        }
