from model import DQN
from agent import Agent
from SnakeGame import Game
from graph import Graph
import torch


def evaluate():
    agent = Agent()
    game = Game(human=False)
    agent.policy_net.load("DQN/models/policy_model_test.pth")
    agent.policy_net.eval()
    graph = Graph()

    rewards = []    # Keep track of the reward earned in each game
    scores = []    # Keep track of the score earned in each game
    num_games = 100
    
    print(f"Evaluating starting for {num_games} games...")

    with torch.no_grad():
        for i_game in range(num_games):
            state = game.reset()
            game_reward = 0
            done = False
            while not done:
                action = agent.get_action(state, use_epsilon=False)[0]
                next_state, reward, running, score = game.play(action)
                done = not running
                if done:
                    next_state = None
                state = next_state
                game_reward += reward

            graph.add_episode(i_game + 1, score)
            scores.append(score)
            rewards.append(game_reward)

            print(f"Game: {i_game}\tReward: {game_reward}\tScore: {score}")
        
        graph.plot()

    print(f"Evaluation completed over {num_games} games.\nAverage Score: {sum(scores)/len(scores)}\nAverage Reward: {sum(rewards)/len(rewards)}")


if __name__ == "__main__":
    evaluate()
