import torch
from model import DQN
from agent import Agent
from SnakeGame import Game

def load_model(path, n_observations, n_actions):
    model = DQN(n_observations, n_actions)
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
    #model = torch.load(path)
    return model

def evaluate():
    agent = Agent()
    game = Game()
    agent.policy_net = load_model("DQN/models/policy_model.pth", 15, 3)
    #agent.policy_net = load_model("DQN/models/policy_model.pth", 15, 3)
    agent.policy_net.eval()
    num_games = 100
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
        print(f"reward: {game_reward} on game {i_game}\n")

if __name__ == "__main__":
    evaluate()