import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        #self.memory = deque([], maxlen=capacity)
        self.regular_memory = deque([], maxlen=int(capacity * 0.8))     # 80% regular
        self.priority_memory = deque([], maxlen=int(capacity * 0.2))    # 20% priority
        self.length_threshold = 0.05    # Starting at length 45 (for a 900 block board)

    def push(self, *args):
        """Save a transition"""
        #self.memory.append(Transition(*args))
        state, action, next_state, reward = args

        # Update the threshold to be 80% of the best length seen
        self.length_threshold = max(self.length_threshold, state[12] * 0.8)

        # Prioritize if snake is long
        if state[12] > self.length_threshold:
            self.priority_memory.append(Transition(*args))
        else:
            self.regular_memory.append(Transition(*args))


    def sample(self, batch_size):
        """Sample a batch of transitions"""
        #return random.sample(self.memory, batch_size)
        priority_size = min(batch_size // 5, len(self.priority_memory))
        regular_size = batch_size - priority_size

        priority_batch = random.sample(list(self.priority_memory), priority_size) if priority_size > 0 else []
        regular_batch = random.sample(list(self.regular_memory), regular_size)

        return priority_batch + regular_batch
    
    def transform(self, batch):
        """Convert a batch of transitions to a single transition of batches"""
        return Transition(*zip(*batch))

    def __len__(self):
        #return len(self.memory)
        return len(self.regular_memory) + len(self.priority_memory)
