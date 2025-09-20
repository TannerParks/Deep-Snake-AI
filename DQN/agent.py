import sys
import os
import random
import numpy as np
import torch
from model import DQN, Trainer
from memory import ReplayMemory
from collections import deque
from colorama import Fore, Back, Style

# Add the path to the SnakeGame.py file to the system path
sys.path.append("../Deep-Snake-AI")
from SnakeGame import Game
from graph import Graph

# Add optuna for hyperparameter optimization
import optuna


class Agent:
    def __init__(self, n_observations=37, n_actions=3, device=torch.device('cuda'),
                 gamma=0.95, learning_rate=0.001, w_decay=0.001,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.games_played = 0
        self.device = device

        # Initialize both policy and target networks
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)

        # Create the Trainer, which includes the optimizer and loss function
        self.trainer = Trainer(self.policy_net, self.target_net, self.device, gamma=gamma, learning_rate=learning_rate,
                               w_decay=w_decay)

        # Use the Trainer's method to initially update the target network
        self.trainer.update_target()

        # Initialize epsilon for epsilon-greedy policy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_max = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_exploitation_phase = False  # Flag to indicate pure exploitation phase
        self.exploitation_games_count = 0
        self.recent_scores = deque([], maxlen=100)
        self.high_score_threshold = 45
        self.epsilon_increase_factor = 1.02
        self.previous_avg_reward = 0
        #self.epsilon_decay_games = 50 # Number of games to linearly decay epsilon to epsilon_min

        print(f"Using device: {self.device}")
        print(f"Policy net device: {next(self.policy_net.parameters()).device}")
        print(f"Target net device: {next(self.target_net.parameters()).device}")

    def update_epsilon_linear(self):
        """Update epsilon based on the number of games played. Linearly decay epsilon to epsilon_min over epsilon_linear_decay games."""
        if self.epsilon > self.epsilon_min:
            # Calculate the decrement after each game
            decrement = (self.epsilon - self.epsilon_min) / self.epsilon_decay_games
            # Decrease epsilon or set it to the minimum value if it's lower than the calculated new epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - decrement)

    def update_epsilon_exponential(self, scores):
        """Update epsilon based on the number of games played with periodic exploitation phases. Starts exploitation only after epsilon is low enough and snake length is good."""

        if len(scores) > 0:
            self.recent_scores.append(scores[-1])  # Track the most recent game score

        # Use a moving average of top 25% of recent scores
        high_scores = sorted(self.recent_scores, reverse=True)[:max(1, len(self.recent_scores) // 4)]
        avg_high_score = np.mean(high_scores) if high_scores else 0.0

        # Start exploitation phases only when conditions are met and the agent has mostly converged on its learned policy
        if self.epsilon <= 0.02 and self.games_played % 200 == 0 and avg_high_score >= self.high_score_threshold and not self.epsilon_exploitation_phase:
            self.epsilon_exploitation_phase = True
            print(f"{Back.LIGHTBLUE_EX}{Fore.BLACK}{Style.BRIGHT}[EPSILON] Exploitation Phase Activated, Average High Score: {avg_high_score}, Old Threshold: {self.high_score_threshold}, New Threshold: {self.high_score_threshold * 1.2}{Style.RESET_ALL}")
            self.high_score_threshold = avg_high_score * 1.2  # Require 20% improvement next time
            self.epsilon = 0

        # Track and end exploitation phase after 5 games
        if self.epsilon_exploitation_phase:
            self.exploitation_games_count += 1
            if self.exploitation_games_count > 5:
                #print("Exploitation Phase Deactivated")
                self.epsilon_exploitation_phase = False
                self.exploitation_games_count = 0

        if not self.epsilon_exploitation_phase:
            #print("Exponential decay")
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(
                -1 * self.games_played / self.epsilon_decay)
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
        is_random = False  # Used to check if a random action led to the death of the snake for analysis purposes

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


def train(agent, num_games, batch_size, tau=0.001, trial=None):
    game = Game(human=False)
    memory = ReplayMemory(100000)
    graph = Graph()
    rewards = []  # Keep track of the reward earned in each game
    scores = []  # Keep track of the score earned in each game
    max_score = 0  # Track the highest score achieved in this training session

    random_deaths = 0

    print(f"Training started for {num_games} games...")

    for i_game in range(num_games):
        state = game.reset()  # Reset the game and get the start state
        game_reward = 0  # Keep track of the reward earned in the current game
        agent.update_epsilon_exponential(scores)
        #print(f"Epsilon: {agent.epsilon}")

        # Play the game until it's over
        done = False
        random_death_flag = False    # Track if random move led to death

        while not done:
            action, was_random = agent.get_action(state)
            next_state, reward, running, score = game.play(action)
            done = not running

            # Setting next_state to None if the game is over will help the agent to recognize the end of the game
            if done:
                if was_random:
                    random_death_flag = True
                    random_deaths += 1
                next_state = None

            # Save the transition in the replay memory
            memory.push(state, action, next_state, reward)

            state = next_state

            # Start training the agent if the memory has enough transitions
            if len(memory) >= batch_size:
                agent.trainer.optimize(memory, batch_size)

            agent.trainer.soft_update_target(tau)

            game_reward += reward

        agent.games_played += 1
        graph.add_episode(agent.games_played, score)
        scores.append(score)
        rewards.append(game_reward)

        game.log_game_stats(timeout=game.process_took_too_long_flag, random_death=random_death_flag, total_reward=game_reward)

        # Update max score for the entire training session
        if score > max_score:
            max_score = score
        #print(f"Game: {i_game}\t\tScore: {score}\t\tReward: {game_reward:.2f}\t\tMax Score: {max_score}\t\tEpsilon: {agent.epsilon:.5f}\t\tRandom Death: {was_random}\n")

        # Optuna Pruning - Report Intermediate Score After every 105 games (Exploitation phases periodically happen in this range too)
        if i_game >= 100 and i_game % 100 == 5:
            mean_score = np.mean(scores[-100:])  # Last 100 games' average score
            best_scores = sorted(scores, reverse=True)[:50]  # FIXME: OPTUNA - Take top 50 best scores
            avg_top_score = np.mean(best_scores)  # FIXME: OPTUNA
            weighted_score = (0.7 * avg_top_score) + (0.3 * max_score)

            if trial is not None:
                print(
                    f"{Back.LIGHTGREEN_EX}{Fore.BLACK}{Style.BRIGHT}[Optuna Report] Step: {i_game}, Max Score: {max_score}, Mean Last 100: {mean_score}, Max Last 100: {max(scores[-100:])}, Top 50 Mean: {avg_top_score}, Weighted Score: {weighted_score}, Reward: {game_reward}{Style.RESET_ALL}")

                trial.report(weighted_score, i_game)

                if trial.should_prune():
                    print(
                        f"{Back.RED}{Fore.BLACK}{Style.BRIGHT}[Optuna] Trial pruned at step {i_game} (Max Score: {max_score}, Mean Last 100: {mean_score}, Max Last 100: {max(scores[-100:])}, Top 50 Mean: {avg_top_score}, Weighted Score: {weighted_score}, Reward: {game_reward}){Style.RESET_ALL}")
                    raise optuna.TrialPruned()
            else:
                print(f"{Back.LIGHTGREEN_EX}{Fore.BLACK}{Style.BRIGHT}[Step Report] Step: {i_game}, Max Score: {max_score}, Mean Last 100: {mean_score}, Max Last 100: {max(scores[-100:])}, Top 50 Mean: {avg_top_score}, Weighted Score: {weighted_score}, Reward: {game_reward}{Style.RESET_ALL}")

    agent.policy_net.save("DQN/models/policy_model.pth")  # FIXME: Comment out for Optuna trials
    
    average_score = np.mean(scores)
    best_scores = sorted(scores, reverse=True)[:50]  # FIXME: OPTUNA - Take top 50 best scores
    avg_top_score = np.mean(best_scores)  # FIXME: OPTUNA
    weighted_score = (0.7 * avg_top_score) + (0.3 * max_score)

    # Print all values and calculations that would be helpful for evaluating the training session
    print(f"{Back.MAGENTA}{Fore.BLACK}{Style.BRIGHT}Training completed over {num_games} games.\nHighest Score: {max_score}\nAverage Score: {average_score}\nTop 50 Average Score: {avg_top_score}\nDeaths by Random Move: {random_deaths}\nWeighted Score: {weighted_score}{Style.RESET_ALL}")
    
    graph.plot()

    #return (0.7 * avg_top_score) + (0.3 * max_score)  # FIXME: OPTUNA


"""OPTUNA HYPERPARAMETER TRAINING"""


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.005, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    #epsilon_start = trial.suggest_float('epsilon_start', 0.8, 1.0)
    #epsilon_end = trial.suggest_float('epsilon_end', 0.001, 0.01)
    epsilon_decay = trial.suggest_int('epsilon_decay', 75, 200)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    tau = trial.suggest_float("tau", 0.001, 0.05)
    w_decay = trial.suggest_float("w_decay", 0.0001, 0.1)

    print(
        f"learning_rate={learning_rate}, gamma={gamma}, epsilon_decay={epsilon_decay}, batch_size={batch_size}, tau={tau}, w_decay={w_decay}")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        dev = torch.device("cpu")

    agent = Agent(learning_rate=learning_rate, gamma=gamma, w_decay=w_decay, epsilon_start=0.9940711467016862, epsilon_end=0.0001,
                  epsilon_decay=epsilon_decay, device=dev)

    return train(agent, num_games=2005, batch_size=batch_size, tau=tau, trial=trial)


def optimize():
    # Get the directory of the current file (agent.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "snake_dqn_study_hyperparameters2.db")

    study = optuna.create_study(study_name="snake_dqn_study_hyperparameters2", storage=f"sqlite:///{db_path}", direction="maximize",
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1405))

    study.enqueue_trial({
       "learning_rate": 0.00036897789293035725,
       "gamma": 0.9852852928197687,
       "epsilon_decay": 150,
       "batch_size": 128,
       "tau": 0.001,
       "w_decay": 0.001
    })

    # Best hyperparameters that I want to improve
    if len(study.trials) == 0:
        study.enqueue_trial({
            "learning_rate": 0.00031827012284614444,
            "gamma": 0.9938604472962519,
            "epsilon_decay": 86,
            "batch_size": 64,
            "tau": 0.0010397946336198054,
            "w_decay": 0.06242709079561533
        })
        #study.enqueue_trial({
        #    "learning_rate": 0.00036897789293035725,
        #    "gamma": 0.9852852928197687,
        #    "epsilon_decay": 150,
        #    "batch_size": 128,
        #    "tau": 0.001,
        #    "w_decay": 0.001
        #})

    study.optimize(objective, n_trials=50)

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print(f"Accuracy: {trial.value}")
    print(f"Best hyperparameters: {trial.params}")


# {'learning_rate': 0.00045978908864575236, 'gamma': 0.9939136048282744, 'epsilon_decay': 144, 'batch_size': 64, 'tau': 0.002015749479720252}

def main():
    #optimize()  # FIXME: OPTUNA Uncomment to run hyperparameter optimization
    best_params = {'learning_rate': 0.000102248605253763, 'gamma': 0.975624424100652, 'w_decay': 0.00940121113083092, 'epsilon_start': 0.9940711467016862, 'epsilon_end': 0.0001, 'epsilon_decay': 90}
    best_batch_size = 128
    best_tau = 0.00781269243820524

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        dev = torch.device("cpu")

    agent = Agent(**best_params, device=dev)
    train(agent, 2000, batch_size=best_batch_size, tau=best_tau)


if __name__ == "__main__":
    main()
