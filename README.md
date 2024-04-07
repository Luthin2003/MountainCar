# MountainCar Q-Learning

## Features

- **Q-Learning Algorithm:** The agent learns to navigate the MountainCar environment using the Q-learning algorithm, which is a model-free reinforcement learning technique.
  
- **Epsilon-Greedy Exploration:** The agent balances exploration and exploitation using an epsilon-greedy strategy, gradually decaying the exploration rate over time.
  
- **Discrete State Space:** To handle the continuous state space of the environment, it's discretized into bins, allowing the agent to learn a Q-value for each state-action pair.
  
- **Visualization:** The environment is rendered periodically to visualize the agent's behavior and progress toward solving the task.

## Files

- `qlearning.py`: Contains the main implementation of the Q-learning algorithm for the MountainCar environment.

## Requirements

- Python 3
- Gym
- NumPy

## Usage

To run the Q-learning agent on the MountainCar environment, simply execute the `mountaincar_q_learning.py` script. You can adjust hyperparameters and other settings in the script as needed.
