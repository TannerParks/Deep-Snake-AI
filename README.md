# Deep Q-Network for the Snake Game: Project Overview

This project focuses on developing a Deep Q-Network (DQN) AI agent capable of learning and playing the classic Snake game (the game portion I use is a tweaked and heavily decoupled version of a Snake project I did in an Intro to CS class). The goal is to create an intelligent agent that can navigate the game environment, make strategic decisions, and achieve high scores by eating fruits while avoiding collisions. This project also includes a graph in order to help visualize the learning process, and also an evaluation tool to test out any saved models. Below is a detailed overview of the project, highlighting key aspects such as state representation, actions, rewards/penalties, and the methodologies employed.

## State Representation
The state representation is foundational to the DQN's decision-making process, encapsulating the game environment at any given moment. It informs the AI agent of its current situation, aiding in determining the next best action. Through experimentation, the project evolved from simple binary matrices to more nuanced continuous values, enhancing the complexity and capability of the neural network. The final state representation includes:

* Direction of Snake: Encoded as binary values indicating the current movement direction (left, right, up, or down).
* Snake to Fruit Direction: Indicates the direction to the fruit from the snake's head, allowing for diagonal representations (e.g., up + right).
* Distance to Fruit: The straight-line distance from the snake's head to the fruit.
* Distance to Nearest Collision: Calculates the proximity to the nearest obstacle (walls or the snake's body) in three relative directions from the snake's head (straight, left, right).
* Length of the Snake: Represents the current length of the snake.
* Accessible Area: The available space in each relative direction, providing insight into potential paths and dead ends.

Normalization is applied to all values to ensure a consistent scale (0 to 1), facilitating efficient and stable optimization during neural network training.

## Actions
The AI agent has three possible actions to navigate the game environment:

* Keep going straight: [1, 0, 0]

* Move left: [0, 1, 0]

* Move right: [0, 0, 1]

This simplified action space allows the AI to make quick, strategic decisions to maneuver the snake effectively.

## Rewards/Penalties
A critical feedback loop for the AI, rewards, and penalties reinforce desirable behaviors and deter unfavorable actions. The AI learns over time to maximize its score by balancing the pursuit of rewards with the avoidance of penalties. The reward system includes:

* Eating a Fruit: +75 points for consuming a fruit, encouraging growth.

* Collision: -75 points for hitting the wall or the snake's body, penalizing failure.

* Distance to Fruit: +1 for moving closer, -1 for moving away, subtly guiding the snake towards the fruit.

* Time Penalty: -10 for taking excessive time to find a fruit, promoting efficiency.

* Tight Spaces: Up to -25 for entering areas with less space than the snake's length, discouraging risky maneuvers. This penalty deactivates when the snake occupies more than half the board.

## Methodology and Insights
* Balanced Learning Approach: Careful balancing of rewards and penalties ensures the AI prioritizes fruit collection without resorting to suboptimal or risky strategies.
* Adaptive Exploration: Dynamically adjusting the exploration rate (epsilon) based on performance, with periods of exploitation to leverage learned behaviors and exploration to discover new strategies. (the currently active epsilon decay strategy is an exponential decay function but it can be switched out with the linear or adaptive strategy, updating either by frame or by game)
* Normalization: Ensures efficient learning by keeping input scales consistent, aiding the neural network's optimization process.
* Continuous Improvement: Regular evaluation and adjustment of the state representation, reward system, and exploration strategy based on the AI's performance and learning progress.

## Conclusion
This project showcases the design of a state representation that captures environmental cues, creating a rewarding system that motivates intelligent behavior, and developing a neural network capable of learning and adapting to the challenges of the Snake game. Through careful experimentation and iterative refinement, the AI agent learns to navigate the game environment with increasing proficiency, demonstrating the potential of DQN in mastering complex tasks. An early version of this project with less complex states and rewards would typically start to level off at about a length of 60 when evaluated. The highest I've seen this one react is a length of 250. While that's pretty good for an AI that doesn't have a complete view of the board, I might revisit it at some later date and see how much it can be improved with things like more detailed state representations and a learning curriculum to teach it things like Hamiltonian Paths.




