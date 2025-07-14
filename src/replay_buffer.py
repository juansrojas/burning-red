import numpy as np
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, terminal)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminal = map(np.stack, zip(*batch))
        return state, action, reward.reshape(-1, 1), next_state, terminal.reshape(-1, 1)

    def __len__(self):
        return len(self.buffer)