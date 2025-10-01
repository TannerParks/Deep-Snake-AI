import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self):
        self.episodes = []
        self.scores = []

    def add_episode(self, episode, score):
        self.episodes.append(episode)
        self.scores.append(score)

    def plot(self):
        plt.figure(figsize=(10, 5))  # Makes the figure larger for better readability
        
        # Actual scores
        plt.plot(self.episodes, self.scores, label='Score', alpha=0.5)
        
        # Calculate and plot the moving average
        window_size = 100
        moving_avg = np.convolve(self.scores, np.ones(window_size)/window_size, mode='valid')
        plt.plot(self.episodes[:len(moving_avg)], moving_avg, label='Moving Average', color='red')
        
        plt.title('DQN Snake Performance')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.ylim(ymin=0, ymax=(max(self.scores) + 10))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    graph = Graph()
    # Simulate adding episodes with varying scores
    for episode in range(1, 101):
        # Example score pattern: start low, increase, fluctuate, then stabilize
        if episode <= 20:
            score = np.random.randint(1, 15)  # Early learning phase, lower scores
        elif episode <= 50:
            score = np.random.randint(10, 50)  # Improvement phase
        elif episode <= 80:
            score = np.random.randint(20, 70)  # Fluctuating phase
        else:
            score = np.random.randint(50, 75)  # Stabilizing phase
        graph.add_episode(episode, score)
    graph.plot()
