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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self): # TODO?: Add parameters to the constructor like device, state_size, action_size, etc.
        self.games_played = 0

        # Initialize both policy and target networks
        self.policy_net = DQN(11, 3).to(device)
        self.target_net = DQN(11, 3).to(device)

        # Create the Trainer, which includes the optimizer and loss function
        self.trainer = Trainer(self.policy_net, self.target_net, gamma=0.99, learning_rate=0.001)

        # Use the Trainer's method to initially update the target network
        self.trainer.update_target()

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_games = 5 # Number of games to linearly decay epsilon to epsilon_min

        # TODO: Implement adaptive epsilon adjustment
        self.epsilon_decay = 0.995
        self.epsilon_adaptive_adjustment = 0.01

    def update_epsilon(self):
        """Update epsilon based on the number of games played. Linearly decay epsilon to epsilon_min over epsilon_linear_decay games."""
        if self.epsilon > self.epsilon_min:
            # Calculate the decrement after each game
            decrement = (self.epsilon - self.epsilon_min) / self.epsilon_decay_games
            # Decrease epsilon or set it to the minimum value if it's lower than the calculated new epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - decrement)
        
    def get_action(self, state):
        """Decide whether to take a random action (explore) or the best action according to the current policy (exploit), based on epsilon."""
        action = [0, 0, 0]

        print(self.policy_net(state))
        print(self.policy_net(state).max(1)[1].view(1, 1).item())
        # If the random number is less than epsilon, randomly choose an index to set to 1 (left, right, or straight)
        if np.random.rand() <= self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            action[0] = 1   # TODO: Implement the best action according to the current policy

        return action   # Might need to convert this to a tensor then get the item



def train():
    agent = Agent()
    game = Game()
    memory = ReplayMemory(10000)
    num_games = 5

    #if torch.cuda.is_available():
    #    print("Using GPU")
    #    num_games = 10  # TODO: Change both num_games to higher numbers
    #else:
    #    print("Using CPU")
    #    num_games = 5


    for i_game in range(num_games):
        state = game.reset()    # Reset the game and get the start state
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)  # Convert the state to a tensor and add a batch dimension
        done = False

        # Play the game until it's over
        while not done:
            agent.update_epsilon()
            action = agent.get_action(state)
            next_state, reward, running, score = game.play(action)
            reward = torch.tensor([reward], device=device, dtype=torch.float32).unsqueeze(0)  # Convert the reward to a tensor and add a batch dimension (might not be necessary) TODO: Check if this is necessaryq
            done = not running

            # Setting next_state to None if the game is over will help the agent to recognize the end of the game
            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

            # Save the transition in the replay memory
            memory.push(state, action, next_state, reward, running)
        
        agent.games_played += 1
                
            


 
            
        

if __name__ == "__main__":
    train()
