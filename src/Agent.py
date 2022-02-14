import numpy as np
import torch as T
import random
from DeepQNetwork import DeepQNetwork

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.replace_target = 100
        self.Q_eval = DeepQNetwork(lr, input_dims, 64, 64 * 2, n_actions)
        self.state_memory = np.zeros((max_mem_size, input_dims))
        self.new_state_memory = np.zeros((max_mem_size, input_dims))
        self.action_memory = np.zeros(max_mem_size)
        self.reward_memory = np.zeros(max_mem_size)
        self.terminal_memory = np.zeros(max_mem_size)

    def store_transition(self, state, action, reward, new_state, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1

    def choose_action(self, observation) -> int:
        n = random.random()
        if n > self.epsilon:
            state = T.as_tensor(observation).float()
            action = self.Q_eval.forward(state)
            return T.argmax(action).numpy()
        else:
            return random.sample(self.action_space, 1)[0]

    def learn(self):
        # 1. If not enough data, return
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        # 2. Retreive a batch of data.
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = [self.state_memory[n] for n in batch]
        new_state_batch = [self.new_state_memory[n] for n in batch]
        reward_batch = [self.reward_memory[n] for n in batch]
        terminal_batch = [self.terminal_memory[n] for n in batch]
        action_batch = [self.action_memory[n] for n in batch]

        # 3. Feedforward
        q_eval = self.Q_eval.forward(T.as_tensor(state_batch).float())
        q_next = self.Q_eval.forward(T.as_tensor(new_state_batch).float())
        q_next[terminal_batch] = 0.0

        # 4. Compute the target
        q_target = self.Q_eval.forward(T.as_tensor(state_batch).float()).detach().numpy()
        for i in batch_index:
            if terminal_batch[i]:
                q_target[i, int(action_batch[i])] = reward_batch[i]
            else:
                q_target[i, int(action_batch[i])] = reward_batch[i] + self.gamma * T.max(q_next[i]).item()
        q_target = T.as_tensor(q_target).float()

        # 5. Backward
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min