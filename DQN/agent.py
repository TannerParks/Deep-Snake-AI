import sys
import random
import numpy as np
from collections import deque, namedtuple

import torch
from model import DQN, Trainer
from memory import ReplayMemory

# Add the path to the SnakeGame.py file to the system path
sys.path.append("../Deep-Snake-AI")

from SnakeGame import Game, Point, BLOCK_SIZE
from graph import Graph

import optuna

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, n_observations=16, n_actions=3, device="cuda",
                 gamma=0.95, learning_rate=0.001,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.games_played = 0
        self.device = device

        # Initialize both policy and target networks
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)

        # Create the Trainer, which includes the optimizer and loss function
        self.trainer = Trainer(self.policy_net, self.target_net, self.device, gamma=gamma, learning_rate=learning_rate)

        # Use the Trainer's method to initially update the target network
        self.trainer.update_target()

        # Initialize epsilon for epsilon-greedy policy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_max = epsilon_start
        self.epsilon_decay = epsilon_decay
        #self.epsilon_decay_games = 50 # Number of games to linearly decay epsilon to epsilon_min

    def update_epsilon_linear(self):
        """Update epsilon based on the number of games played. Linearly decay epsilon to epsilon_min over epsilon_linear_decay games."""
        if self.epsilon > self.epsilon_min:
            # Calculate the decrement after each game
            decrement = (self.epsilon - self.epsilon_min) / self.epsilon_decay_games
            # Decrease epsilon or set it to the minimum value if it's lower than the calculated new epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - decrement)
    
    def update_epsilon_exponential(self):
        """Update epsilon based on the number of games played. Exponentially decay epsilon to epsilon_min."""
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1 * self.games_played / self.epsilon_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_action(self, state, use_epsilon=True):
        """Decide whether to take a random action (explore) or the best action according to the current policy (exploit), based on epsilon."""
        action = [0, 0, 0]

        # If the random number is less than epsilon, randomly choose an index to set to 1 (left, right, or straight)
        if use_epsilon and np.random.rand() <= self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)  # TODO: dtype might be int instead of float32
            move = self.policy_net(state_tensor).max(1)[1].view(1, 1).item()

        action[move] = 1
        
        return action



def train(agent, num_games):
    game = Game()
    memory = ReplayMemory(50000)
    graph = Graph()
    BATCH_SIZE = 32
    max_score = 0
    scores = []
    high_score = 0

    print(f"Using: {agent.device}")

    for i_game in range(num_games):
        state = game.reset()    # Reset the game and get the start state
        game_reward = 0        # Keep track of the reward earned in the current game

        # Play the game until it's over
        done = False
        while not done:
            #agent.update_epsilon_linear()
            agent.update_epsilon_exponential()

            action = agent.get_action(state)
            next_state, reward, running, score = game.play(action)
            done = not running

            # Setting next_state to None if the game is over will help the agent to recognize the end of the game
            if done:
                next_state = None

            # Save the transition in the replay memory
            memory.push(state, action, next_state, reward, running)

            state = next_state

            # Start training the agent if the memory has enough transitions
            if len(memory) >= BATCH_SIZE:
                agent.trainer.optimize(memory, BATCH_SIZE)
            
            agent.trainer.soft_update_target(0.001) # TODO: Check if the tau value is correct
            
            game_reward += reward
            high_score = max(high_score, score)
            
        agent.games_played += 1
        max_score = max(max_score, game_reward)
        graph.add_episode(agent.games_played, score)
        scores.append(score)
        print(f"reward: {game_reward} on game {i_game}\n")

    torch.save(agent.policy_net.state_dict(), "DQN/models/policy_model.pth")
    average = np.mean(scores)
    print(f"Max score: {max_score}\nHigh score: {high_score}\nAverage: {average}\nMemory: {len(memory)}")
    graph.plot()

    return average


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.9999, log=True)
    epsilon_start = trial.suggest_uniform('epsilon_start', 0.8, 1.0)
    epsilon_end = trial.suggest_uniform('epsilon_end', 0.01, 0.1)
    epsilon_decay = trial.suggest_loguniform('epsilon_decay', 0.9, 0.999)

    agent = Agent(learning_rate=learning_rate, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay)

    average = train(agent, 100)
    return average

def optimize():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=75)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def main():
    #best_params = {'learning_rate': 0.0019449287803023538, 'gamma': 0.899902784288163, 'epsilon_start': 0.904971615780917, 'epsilon_end': 0.010039349408417542, 'epsilon_decay': 0.9094681595057748}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(device=dev)
    train(agent, 200)

if __name__ == "__main__":
    main()
