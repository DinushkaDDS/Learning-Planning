import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
import numpy as np
import gymnasium as gym


def create_model(number_observation_features: int, number_actions: int) -> nn.Module:
    hidden_layer_features = 512

    return nn.Sequential(
        nn.Linear(in_features=number_observation_features,
                  out_features=hidden_layer_features),
        nn.LeakyReLU(),
        nn.Linear(in_features=hidden_layer_features,
                  out_features=hidden_layer_features//2),
        nn.LeakyReLU(),
        nn.Linear(in_features=hidden_layer_features//2,
                  out_features=number_actions),
    )


def get_policy(model: nn.Module, observation: np.ndarray, device) -> Categorical:
    """
        Get the policy from the model, for a specific observation
    """
    observation = observation / 255.0
    observation_tensor = torch.as_tensor(observation, dtype=torch.float32).to(device)
    logits = model(observation_tensor)

    # Categorical will also normalize the logits for us
    return Categorical(logits=logits)


def get_action(policy: Categorical) -> tuple[int, torch.Tensor]:
    """
        Sample an action from the policy
    """
    action = policy.sample()  # Unit tensor

    # Converts to an int, as this is what Gym environments require
    action_int = int(action.item())

    # Calculate the log probability of the action
    log_probability_action = policy.log_prob(action)

    return action_int, log_probability_action


def calculate_loss(epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor) -> torch.Tensor:
    """
        Calculate the 'loss' required to get the policy gradient
    """
    loss = -(epoch_log_probability_actions * epoch_action_rewards).mean()

    print(loss)
    return loss


def train_one_epoch(
        env: gym.Env,
        model: nn.Module,
        optimizer: Optimizer,
        device,
        num_trajectories=10,
        trajectory_length=2000) -> float:
    """
        Train the model for one epoch
    """
    epoch_total_trajectories= 0

    # Returns from each trajectory (to keep track of progress)
    epoch_returns: list[float] = []

    # Action log probabilities and rewards per step (for calculating loss)
    epoch_log_probability_actions = []
    epoch_action_rewards = []

    while True:
        # Break if number of episodes exceed the given amount
        if epoch_total_trajectories > num_trajectories:
            break

        episode_reward: float = 0
        state, info = env.reset()

        for timestep in range(trajectory_length):

            epoch_total_trajectories += 1

            # Get the policy and act
            policy = get_policy(model, state, device)
            action, log_probability_action = get_action(policy)
            state, reward, done, truncated, _ = env.step(action)

            done = done or truncated

            # Increment the episode rewards
            episode_reward += reward

            # Add epoch action log probabilities
            epoch_log_probability_actions.append(log_probability_action)

            # Finish the action loop if this episode is done
            if done is True:
                # Add one reward per timestep
                for _ in range(timestep + 1):
                    epoch_action_rewards.append(episode_reward)

                break

        # Increment the epoch returns
        epoch_returns.append(episode_reward)

    # Calculate the policy gradient, and use it to step the weights & biases
    epoch_loss = calculate_loss(torch.stack(
        epoch_log_probability_actions),
        torch.as_tensor(epoch_action_rewards, dtype=torch.float32).to(device)
    )
    
    epoch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return float(np.mean(epoch_returns))
