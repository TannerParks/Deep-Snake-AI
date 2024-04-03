from model import DQN
from agent import Agent
from SnakeGame import Game

def evaluate():
    agent = Agent()
    game = Game(human=False)
    agent.policy_net.load("DQN/models/policy_model.pth")
    agent.policy_net.eval()

    rewards = []    # Keep track of the reward earned in each game
    scores = []    # Keep track of the score earned in each game

    num_games = 10
    for i_game in range(num_games):
        state = game.reset()
        game_reward = 0
        done = False
        while not done:
            action = agent.get_action(state, use_epsilon=False)
            next_state, reward, running, score = game.play(action)
            done = not running
            if done:
                next_state = None
            state = next_state
            game_reward += reward

        scores.append(score)
        rewards.append(game_reward)
        print(f"Game: {i_game}\tReward: {game_reward}\tScore: {score}")

if __name__ == "__main__":
    evaluate()