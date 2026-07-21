# Custom Deep Q-Learning VS Snake Game and Cart Pole v1

This repository contains an implementation of Deep Q-Learning (DQN) applied to two environments:
1. **Custom Snake Game**: A custom environment built using `pygame` and `gymnasium`.
2. **Cart Pole v1**: The classic CartPole environment from OpenAI Gym (now Gymnasium).

## Project Structure

- `snake_game.py`: The core mechanics of the Snake game using Pygame. It includes a manual mode that you can play yourself.
- `snake_env.py`: A `gymnasium` wrapper around the Snake game, which allows the Deep Q-Learning agent to interact with it.
- `main-RL-code.ipynb`: A Jupyter Notebook containing the training code and RL architecture for both Cart Pole and the custom Snake game.
- `cart_pole_model.h5`: The trained model weights for Cart Pole.
- `snake_model.h5`: The trained model weights for the Snake game.

## Installation

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

### Play the Snake Game Manually
You can play the custom Snake game yourself by running:
```bash
python snake_game.py --speed 10
```
- Controls: Arrow Keys to move, Space to pause, R to restart, ESC to quit.

### Train the Agent
Open `main-RL-code.ipynb` in Jupyter Notebook or JupyterLab to explore the deep Q-learning implementation, train the agents from scratch, or load the pre-trained weights (`.h5` files) to see them in action.

## Technologies Used
- Reinforcement Learning (Deep Q-Learning)
- TensorFlow / Keras
- Gymnasium
- Pygame
- NumPy
