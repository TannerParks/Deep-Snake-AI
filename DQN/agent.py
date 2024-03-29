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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self): # TODO?: Add parameters to the constructor like device, state_size, action_size, etc.
        self.games_played = 0

        # Initialize both policy and target networks
        self.policy_net = DQN(15, 3).to(device)
        self.target_net = DQN(15, 3).to(device)

        # Create the Trainer, which includes the optimizer and loss function
        self.trainer = Trainer(self.policy_net, self.target_net, device, gamma=0.95, learning_rate=0.001)

        # Use the Trainer's method to initially update the target network
        self.trainer.update_target()

        self.epsilon = 1.0
        self.epsilon_min = 0.01 # TODO: Test different values for epsilon_min (0.01, 0.05, 0.1, etc.)
        self.epsilon_max = 1.0
        self.epsilon_decay_games = 50 # Number of games to linearly decay epsilon to epsilon_min

        # TODO: Implement adaptive epsilon adjustment
        self.epsilon_decay = 0.995
        self.epsilon_adaptive_adjustment = 0.01
        self.epsilon_improvement_threshold = 1.05   # If the average reward for a game is below this threshold, decrease epsilon
        self.previous_score = 0

    def update_epsilon_linear(self):
        """Update epsilon based on the number of games played. Linearly decay epsilon to epsilon_min over epsilon_linear_decay games."""
        if self.epsilon > self.epsilon_min:
            # Calculate the decrement after each game
            decrement = (self.epsilon - self.epsilon_min) / self.epsilon_decay_games
            # Decrease epsilon or set it to the minimum value if it's lower than the calculated new epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - decrement)
    
    def update_epsilon_exponential(self):
        """Update epsilon based on the number of games played. Exponentially decay epsilon to epsilon_min."""
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1. * self.games_played / self.epsilon_decay)
        self.epsilon = max(self.epsilon_min, self.epsilon)
    
    def update_epsilon_adaptive(self, current_score):
        if current_score > self.previous_score * self.epsilon_improvement_threshold:
            # Agent is improving, increase epsilon to encourage exploration
            self.epsilon *= self.epsilon_decay
        else:
            # Agent is not improving, decrease epsilon to encourage exploitation
            self.epsilon += self.epsilon_adaptive_adjustment
        
        self.epsilon = max(self.epsilon_min, min(self.epsilon, self.epsilon_max))

        self.previous_score = current_score

    def get_action(self, state, use_epsilon=True):
        """Decide whether to take a random action (explore) or the best action according to the current policy (exploit), based on epsilon."""
        action = [0, 0, 0]

        # If the random number is less than epsilon, randomly choose an index to set to 1 (left, right, or straight)
        if use_epsilon and np.random.rand() <= self.epsilon:
            move = random.randint(0, 2)
            #print(f"random: {move}")
        else:
            state_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)  # TODO: dtype might be int instead of float32
            move = self.policy_net(state_tensor).max(1)[1].view(1, 1).item()
            #print(f"policy: {move}")

        action[move] = 1
        
        return action



def train():
    agent = Agent()
    game = Game()
    memory = ReplayMemory(10000)
    graph = Graph()
    BATCH_SIZE = 10
    #recent_scores = deque(maxlen=5)  # Keep track of the last 5 scores (reward earned in each game)

    if torch.cuda.is_available():
        print("Using GPU")
        num_games = 200
    else:
        print("Using CPU")
        num_games = 100

    for i_game in range(num_games):
        state = game.reset()    # Reset the game and get the start state
        game_reward = 0        # Keep track of the reward earned in the current game
        #agent.update_epsilon_adaptive(game_reward)

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
            
            agent.trainer.soft_update_target(0.001)
            
            game_reward += reward
            
        agent.games_played += 1
        print(f"reward: {game_reward} on game {i_game}\n")
        #recent_scores.append(game_reward)
        graph.add_episode(agent.games_played, score)
    torch.save(agent.policy_net.state_dict(), "DQN/models/policy_model.pth")   # TODO: don't know if this works
    graph.plot()
                

if __name__ == "__main__":
    train()
