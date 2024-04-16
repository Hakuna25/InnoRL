import torch
import numpy as np
import random
import collections

class MultiAgentReplayBuffer:
    def __init__(self, config, Discrete=True):
        self.mem_size = config.mem_size
        self.mem_cntr = 0
        self.n_agents = config.n_agents
        self.actor_dims = config.state_space
        self.batch_size = config.batch_size
        self.n_actions = config.action_space

        self.n_rollout_threads = config.n_rollout_threads
        self.state_memory = np.zeros((self.mem_size, self.n_rollout_threads, self.observation_space))
        self.new_state_memory = np.zeros((self.mem_size, self.n_rollout_threads, self.observation_space))
        self.reward_memory = np.zeros((self.mem_size, self.n_rollout_threads, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_rollout_threads, self.n_agents), dtype=bool)
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.n_rollout_threads, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.n_rollout_threads, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_rollout_threads, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size
        re = []
        for i, agent in enumerate(raw_obs):
            self.actor_state_memory[i, :, index] = raw_obs[agent]
            self.actor_new_state_memory[i, :, index] = raw_obs_[agent]
            self.actor_action_memory[i, :, index] = action[agent]
            re.append(reward[agent])

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = np.array(re)
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx,:,batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True

class MultiAgentBaseReplayBuffer:
    def __init__(self, config, Discrete=True):
        self.mem_size = config.mem_size
        self.mem_cntr = 0
        self.n_agents = config.n_agents
        self.actor_dims = config.state_space
        self.batch_size = config.batch_size
        self.n_actions = config.action_space
        self.state_memory = np.zeros((self.mem_size, config.observation_space))
        self.new_state_memory = np.zeros((self.mem_size, config.observation_space))
        self.reward_memory = np.zeros((self.mem_size, config.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, config.n_agents), dtype=bool)
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward,
                         raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size
        re = []
        for i, agent in enumerate(raw_obs):
            self.actor_state_memory[i][index] = raw_obs[agent]
            self.actor_new_state_memory[i][index] = raw_obs_[agent]
            self.actor_action_memory[i][index] = action[agent]
            re.append(reward[agent])

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = np.array(re)
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])
        return actor_states, states, actions, rewards, \
               actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True

class OnPolicyBaseReplayBuffer:
    def __init__(self, config, Discrete=True):
        self.s = np.zeros((config.batch_size, config.state_space),dtype=np.float32)
        self.r = np.zeros((config.batch_size, 1),dtype=np.float32)
        self.s_ = np.zeros((config.batch_size, config.state_space),dtype=np.float32)
        self.dw = np.zeros((config.batch_size, 1),dtype=np.bool_)
        self.done = np.zeros((config.batch_size, 1),dtype=np.bool_)
        if Discrete:
            self.a = np.zeros((config.batch_size, 1), dtype=np.int64)
            self.a_logprob = np.zeros((config.batch_size, 1),dtype=np.float32)
        else:
            self.a = np.zeros((config.batch_size, config.action_space), dtype=np.float32)
            self.a_logprob = np.zeros((config.batch_size, config.action_space),dtype=np.float32)
        self.count = 0
        self.Discrete = Discrete

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.from_numpy(self.s)
        r = torch.from_numpy(self.r)
        s_ = torch.from_numpy(self.s_)
        a_logprob = torch.from_numpy(self.a_logprob)
        if self.Discrete:
            dw = torch.tensor(self.dw, dtype=torch.float)
            done = torch.tensor(self.done, dtype=torch.float)
            a = torch.tensor(self.a, dtype=torch.long)
        else:
            dw = torch.from_numpy(self.dw)
            done = torch.from_numpy(self.done)
            a = torch.from_numpy(self.a)
        return s, a, a_logprob, r, s_, dw, done

class OnPolicyReplayBuffer:
    def __init__(self, config, Discrete=True):
        self.s = np.zeros((config.batch_size, config.n_rollout_threads, config.n_agents, config.state_space), dtype=np.float32)
        self.r = np.zeros((config.batch_size, config.n_rollout_threads, 1), dtype=np.float32)
        self.s_ = np.zeros((config.batch_size, config.n_rollout_threads, config.n_agents, config.state_space), dtype=np.float32)
        self.dw = np.zeros((config.batch_size, config.n_rollout_threads,  1), dtype=np.bool_)
        self.done = np.zeros((config.batch_size, config.n_rollout_threads,  1), dtype=np.bool_)
        if Discrete:
            self.a = np.zeros((config.batch_size, config.n_rollout_threads, config.n_agents, 1), dtype=np.int64)
            self.a_logprob = np.zeros((config.batch_size, config.n_rollout_threads, config.n_agents, 1), dtype=np.float32)
        else:
            self.a = np.zeros((config.batch_size, config.n_rollout_threads, config.n_agents, config.action_space), dtype=np.float32)
            self.a_logprob = np.zeros((config.batch_size, config.n_rollout_threads, config.n_agents, config.action_space), dtype=np.float32)
        self.count = 0
        self.Discrete = Discrete
        self.n_agnets = 1

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.s_[self.count] = s_

        self.dw[self.count] = dw
        self.done[self.count] = done
        self.r[self.count] = np.reshape(r, (r.shape[0], 1))
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.from_numpy(self.s)
        r = torch.from_numpy(self.r)
        s_ = torch.from_numpy(self.s_)
        a_logprob = torch.from_numpy(self.a_logprob)
        if self.Discrete:
            dw = torch.tensor(self.dw, dtype=torch.float)
            done = torch.tensor(self.done, dtype=torch.float)
            a = torch.tensor(self.a, dtype=torch.long)
        else:
            dw = torch.from_numpy(self.dw)
            done = torch.from_numpy(self.done)
            a = torch.from_numpy(self.a)
        return s, a, a_logprob, r, s_, dw, done


class Offpolicy_ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)