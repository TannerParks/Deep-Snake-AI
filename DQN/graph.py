import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.episodes = []
        self.scores = []

    def add_episode(self, episode, score):
        self.episodes.append(episode)
        self.scores.append(score)

    def plot(self):
        plt.plot(self.episodes, self.scores)
        plt.title('DQN Performance')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.ylim(ymin=0)
        plt.show()


if __name__ == "__main__":
    graph = Graph()
    graph.add_episode(1, 10)
    graph.add_episode(2, 20)
    graph.add_episode(3, 30)
    graph.plot()