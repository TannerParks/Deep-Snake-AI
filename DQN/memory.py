import random
import numpy as np
from collections import namedtuple, deque
import heapq

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.regular_memory = deque([], maxlen=int(capacity * 0.8))  # 80% regular
        self.priority_memory = deque([], maxlen=int(capacity * 0.2))  # 20% priority
        self.length_threshold = 0.05  # Starting at length ~45 (for a 900 block board)
        self.high_scores_heap = []

    def push(self, *args):
        """Save a transition"""
        state, action, next_state, reward = args
        current_length = state[12]  # Normalized length (0-1)

        # Only store in the min-heap if we have room OR it's higher than the lowest top score
        if len(self.high_scores_heap) < max(1, len(self.priority_memory) // 10):
            heapq.heappush(self.high_scores_heap, current_length)
        elif current_length > self.high_scores_heap[0]:  # Only push if better than smallest in heap
            heapq.heappushpop(self.high_scores_heap, current_length)

        # Update length threshold based on 75% of the mean of the top scores
        if self.high_scores_heap:
            self.length_threshold = max(self.length_threshold, np.mean(self.high_scores_heap) * 0.75)

        # print(f"[MEMORY] New Length: {current_length} or {current_length * 900}| Threshold: {self.length_threshold} or {self.length_threshold * 900}")
        # print(f"[MEMORY] Heap Contents (Top Scores): {sorted(self.high_scores_heap, reverse=True)}")

        # Prioritize if snake is long
        if current_length > self.length_threshold:
            self.priority_memory.append(Transition(*args))
            # print(f"[MEMORY] Stored in PRIORITY memory (>{self.length_threshold} or {self.length_threshold * 900})")
        else:
            self.regular_memory.append(Transition(*args))
            # print(f"[MEMORY] Stored in REGULAR memory")

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        priority_size = min(batch_size // 5, len(self.priority_memory))
        regular_size = batch_size - priority_size

        priority_batch = random.sample(list(self.priority_memory), priority_size) if priority_size > 0 else []
        regular_batch = random.sample(list(self.regular_memory), regular_size)

        return priority_batch + regular_batch

    def transform(self, batch):
        """Convert a batch of transitions to a single transition of batches"""
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.regular_memory) + len(self.priority_memory)

    def transform(self, batch):
        """Convert a batch of transitions to a single transition of batches"""
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.regular_memory) + len(self.priority_memory)
