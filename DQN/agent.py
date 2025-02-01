import sys
import random
import numpy as np
import torch
from model import DQN, Trainer
from memory import ReplayMemory

# Add the path to the SnakeGame.py file to the system path
sys.path.append("../Deep-Snake-AI")
from SnakeGame import Game
from graph import Graph

# Add optuna for hyperparameter optimization
import optuna


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
        self.epsilon_increase_factor = 1.02
        self.previous_avg_reward = 0
        self.epsilon_exploitation_phase = False # Flag to indicate pure exploitation phase
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
        if self.games_played >= 100 and self.games_played % 100 == 0:
            self.epsilon_exploitation_phase = True
            self.epsilon = 0
        elif self.games_played >= 100 and self.games_played % 100 == 5:
            self.epsilon_exploitation_phase = False

        if not self.epsilon_exploitation_phase:
            #print("Exponential decay")
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1 * self.games_played / self.epsilon_decay)
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def update_epsilon_adaptive(self, rewards):
        recent_avg_reward = np.mean(rewards[-10:])
        epsilon_threshold = 0.001
        if recent_avg_reward > self.previous_avg_reward * 1.05:
            if self.epsilon == 0:
                self.epsilon = 0.2
            else:
                self.epsilon = min(self.epsilon * self.epsilon_increase_factor, self.epsilon_max)
            print(f"EXPLORATION:\t\t{self.epsilon}")
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            #self.update_epsilon_exponential()
            if self.epsilon < epsilon_threshold:
                self.epsilon = 0
            print(f"STANDARD:\t\t{self.epsilon}")
        
        self.previous_avg_reward = recent_avg_reward

    def get_action(self, state, use_epsilon=True):
        """Decide whether to take a random action (explore) or the best action according to the current policy (exploit), based on epsilon."""
        action = [0, 0, 0]
        is_random = False   # Used to check if a random action led to the death of the snake for analysis purposes

        # If the random number is less than epsilon, randomly choose an index to set to 1 (left, right, or straight)
        if use_epsilon and np.random.rand() <= self.epsilon:
            move = random.randint(0, 2)
            is_random = True
        else:
            state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            move = q_values.max(1)[1].view(1, 1).item()

        action[move] = 1
        
        return action, is_random



def train(agent, num_games):
    game = Game(human=False)
    memory = ReplayMemory(100000)
    graph = Graph()
    BATCH_SIZE = 128
    rewards = []    # Keep track of the reward earned in each game
    scores = []    # Keep track of the score earned in each game

    random_deaths = 0

    print(f"Using: {agent.device}")

    for i_game in range(num_games):
        state = game.reset()    # Reset the game and get the start state
        game_reward = 0        # Keep track of the reward earned in the current game

        #agent.update_epsilon_adaptation(rewards)
        agent.update_epsilon_exponential()

        print(f"Epsilon: {agent.epsilon}")

        # Play the game until it's over
        done = False
        while not done:
            #agent.update_epsilon_exponential()

            action, was_random = agent.get_action(state)
            next_state, reward, running, score = game.play(action)
            done = not running

            # Setting next_state to None if the game is over will help the agent to recognize the end of the game
            if done:
                if was_random:
                    print("Death caused by random action")
                    random_deaths += 1
                next_state = None

            # Save the transition in the replay memory
            memory.push(state, action, next_state, reward)

            state = next_state

            # Start training the agent if the memory has enough transitions
            if len(memory) >= BATCH_SIZE:
                agent.trainer.optimize(memory, BATCH_SIZE)
            
            agent.trainer.soft_update_target(0.001)
            
            game_reward += reward
            
        agent.games_played += 1
        graph.add_episode(agent.games_played, score)
        scores.append(score)
        rewards.append(game_reward)
        print(f"Game: {i_game}\t\t\tScore: {score}\t\t\tReward: {game_reward}\n")

    agent.policy_net.save("./models/policy_model.pth")
    average_score = np.mean(scores)
    print(f"Highest reward: {max(rewards)}\nHighest score: {max(scores)}\nAverage score: {average_score}\nDeaths by Random Move: {random_deaths}")
    graph.plot()

    return average_score


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.001, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    epsilon_start = trial.suggest_float('epsilon_start', 0.8, 1.0)
    epsilon_end = trial.suggest_float('epsilon_end', 0.001, 0.01)
    epsilon_decay = trial.suggest_float('epsilon_decay', 50, 200)

    agent = Agent(learning_rate=learning_rate, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay)

    average_score = train(agent, 150)
    return average_score

def optimize():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print(f"Accuracy: {trial.value}")
    print(f"Best hyperparameters: {trial.params}")

def main():
    #optimize() # Uncomment to run hyperparameter optimization
    best_params1 = {'learning_rate': 0.00036897789293035725, 'gamma': 0.9852852928197687, 'epsilon_start': 0.9940711467016862, 'epsilon_end': 0.010497463030421313, 'epsilon_decay': 0.9788761529718399}
    best_params2 = {'learning_rate': 0.00036897789293035725, 'gamma': 0.9852852928197687, 'epsilon_start': 0.9940711467016862, 'epsilon_end': 0.001, 'epsilon_decay': 0.9788761529718399}
    best_params3 = {'learning_rate': 0.00036897789293035725, 'gamma': 0.9852852928197687, 'epsilon_start': 0.9940711467016862, 'epsilon_end': 0.001, 'epsilon_decay': 150}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(**best_params3, device=dev)
    train(agent, 2000)

if __name__ == "__main__":
    main()
