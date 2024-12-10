import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import defaultdict

class DynamicGridWorldEnv(gym.Env):
    """Custom Environment for the Dynamic Grid World."""

    def __init__(self, prize_spawn_prob=0.5, repair_shop_pos = None,
                  monster_spawn_prob = 0.5, num_possible_monster_points = 5, possible_monster_pos = []):
        super(DynamicGridWorldEnv, self).__init__()
        print("Top Right of the grid considered as (0,0) coordinate in the renders!")
        print()
        
        # Grid dimensions
        self.grid_size = 5
        self.prize_spawn_prob = prize_spawn_prob

        # For the environment object we fix the repair station location
        if(repair_shop_pos == None):
            self.repair_station = [random.randint(0, 4), random.randint(0, 4)]  # Fixed random repair station position
        else:
            self.repair_station = repair_shop_pos

        self.monster_spawn_prob  = monster_spawn_prob

        # For environment monster spawning areas are fixed.
        if(possible_monster_pos == []):
            
            if(num_possible_monster_points == None):
                # can half of positions be monster spawning points
                num_monsters_pos = random.randint(1, 15)
            else:
                num_monsters_pos = num_possible_monster_points

            self.possible_monster_positions = []
            for _ in range(num_monsters_pos):
                pos = [random.randint(0, 4), random.randint(0, 4)]
                while(pos == self.repair_station):
                    pos = [random.randint(0, 4), random.randint(0, 4)]
                self.possible_monster_positions.append(pos)
        else:
            self.possible_monster_positions = possible_monster_pos


        # Observation space: (X, Y, P, D)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size),  # X coordinate
            spaces.Discrete(self.grid_size),  # Y coordinate
            spaces.Discrete(5),               # P: Prize location (0-4)
            spaces.Discrete(2)                # D: Damaged (0/1)
        ))

        self.prize_positions = {
                2: [0, 0], # bottom-left
                3: [0, 4], # bottom-right
                0: [4, 0], # top-left
                1: [4, 4]  # top-right
            }
        
        self.action_map = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right"
        }

        # Action space: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # Initialize environment parameters
        self.reset()

    def reset(self):
        """Resets the environment to an initial state."""
        self.agent_pos = [random.randint(0, 4), random.randint(0, 4)]
        self.prize_location = random.randint(0, 4)  # 0-4, where 4 means no prize
        self.damaged = 0
        self.monster_positions = []

        return self._get_obs()

    def _get_obs(self):
        """Returns the current state."""
        x, y = self.agent_pos
        return (x, y, self.prize_location, self.damaged)

    def step(self, action):
        """Executes an action and updates the environment."""
        assert self.action_space.contains(action), f"Invalid action {action}"

        x, y = self.agent_pos

        # Movement logic (Actions are handled as we see, not based on coordinate values)
        if action == 0 and y > 0:  # Up
            self.agent_pos[1] -= 1
        elif action == 1 and y < self.grid_size - 1:  # Down
            self.agent_pos[1] += 1
        elif action == 2 and x > 0:  # Left
            self.agent_pos[0] -= 1
        elif action == 3 and x < self.grid_size - 1:  # Right
            self.agent_pos[0] += 1
        else:
            reward = -1  # Hitting a wall
            return self._get_obs(), reward, False, {}

        # Check for interactions with prizes
        reward = 0
        if self.prize_location != 4:
            prize_pos = self.prize_positions[self.prize_location]

            if self.agent_pos == prize_pos:
                reward += 10
                self.prize_location = 4  # Remove the prize

        # Monster interaction
        if self.agent_pos in self.monster_positions:
            self.damaged = 1

        # Repair interaction
        if self.agent_pos == self.repair_station:
            self.damaged = 0

        # Damage penalty
        if self.damaged:
            reward -= 10

        # Randomly respawn prize
        if self.prize_location == 4 and random.random() < self.prize_spawn_prob:
            self.prize_location = random.randint(0, 3)

        # Update monster positions
        self._update_monsters()

        return self._get_obs(), reward, False, {}

    def _update_monsters(self):
        """Randomly update monster positions."""
        self.monster_positions = []
        for pos in (self.possible_monster_positions):

            if(random.random() > self.monster_spawn_prob):
                continue

            # Monsters cannot be spawn in other point of interest locations
            if(self.prize_location!=4):
                if(pos == self.prize_positions[self.prize_location]):
                    continue

            self.monster_positions.append(pos)

    def render(self):
        """Renders the grid world."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid.fill(".")  # Empty cell

        # Place the agent
        x, y = self.agent_pos
        grid[y, x] = "A"

        # Place the prize
        if self.prize_location != 4:
            px, py = self.prize_positions[self.prize_location]
            grid[py, px] = "P"

        # Place monsters
        for mx, my in self.monster_positions:
            grid[my, mx] = "M"

        # Place repair station
        rx, ry = self.repair_station
        grid[ry, rx] = "R"

        print("\n".join(" ".join(row) for row in grid))
        # if(self.prize_location != 4):
        #     print(f"Agent Pos:{self.agent_pos}, Monster Pos:{self.monster_positions},  Repair Pos:{self.repair_station}, Prize Pos:{self.prize_positions[self.prize_location]}")
        # else:
        #     print(f"Agent Pos:{self.agent_pos}, Monster Pos:{self.monster_positions},  Repair Pos:{self.repair_station}, Prize Pos:{[]}")
        print()

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, state_space, action_space, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = defaultdict(lambda: np.zeros(action_space))  # Q-value table
        self.alpha = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon
        self.state_space = state_space
        self.action_space = action_space

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning update rule."""

        # Find best action
        best_next_action = np.argmax(self.q_table[next_state])
        
        # Calculate the Q value
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]

        # Difference between current estimate and calculated one
        td_error = td_target - self.q_table[state][action]

        # Update the table
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# Training the agent
def train_agent(env, agent, episodes=1000, max_steps=100):
    rewards = []  # List to store cumulative rewards per episode

    for episode in range(episodes):
        state = env.reset()  # Get initial state
        state = tuple(state)  # Convert state to tuple for Q-table lookup
        cumulative_reward = 0

        for _ in range(max_steps):
            action = agent.choose_action(state)  # Choose an action
            next_state, reward, done, _ = env.step(action)  # Take the action
            next_state = tuple(next_state)  # Convert to tuple for Q-table lookup

            # Update the Q-table
            agent.update(state, action, reward, next_state)

            state = next_state
            cumulative_reward += reward

            if done:
                break

        # Store cumulative rewards and decay epsilon
        rewards.append(cumulative_reward)
        agent.decay_epsilon()

        # Log progress every 100 episodes
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Cumulative Reward: {cumulative_reward}, Epsilon: {agent.epsilon:.3f}")

    return rewards

# Test the trained policy
def test_agent(env, agent, episodes=10, max_steps=20, render=True):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        state = tuple(state)
        cumulative_reward = 0

        print(f"Episode {episode + 1}")
        env.render()

        for _ in range(max_steps):
            action = np.argmax(agent.q_table[state])  # Exploit the learned policy
            next_state, reward, done, _ = env.step(action)
            state = tuple(next_state)
            cumulative_reward += reward
            if(render):
                env.render()

            if done:
                break
        print(f"Cumulative Reward for test: {cumulative_reward}\n")
        rewards.append(cumulative_reward)

    return rewards


def test_approx_agent(env, agent, episodes=10, max_steps=20, render=True):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        state = tuple(state)
        cumulative_reward = 0

        print(f"Episode {episode + 1}")
        env.render()

        for _ in range(max_steps):
            q_values = [agent.get_q_value(state, action) for action in range(agent.action_space)]
            action =  np.argmax(q_values)
            next_state, reward, done, _ = env.step(action)
            state = tuple(next_state)
            cumulative_reward += reward
            if(render):
                env.render()

            if done:
                break
        print(f"Cumulative Reward for test: {cumulative_reward}\n")
        rewards.append(cumulative_reward)

    return rewards

class ApproxQLearningAgent:
    def __init__(self, feature_extractor, env, action_space, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.weights = np.ones(10)  # Initialize weights
        self.feature_extractor = feature_extractor
        self.env = env # Only using to access the position mappings
        self.action_space = action_space
        self.alpha = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon

    def get_q_value(self, state, action):
        """Compute the Q-value for a given state-action pair."""
        features = self.feature_extractor(self.env, state, action)
        return np.dot(self.weights, features)

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        q_values = [self.get_q_value(state, action) for action in range(self.action_space)]
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state):
        """Update weights using the Q-learning update rule."""
        features = self.feature_extractor(self.env, state, action)
        next_q_values = [self.get_q_value(next_state, a) for a in range(self.action_space)]
        
        target = reward + self.gamma * max(next_q_values)
        td_error = target - self.get_q_value(state, action)
        self.weights += self.alpha * td_error * features

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def feature_extractor(env, state, action):
    """Extract features for approximate Q-learning."""
    agent_x, agent_y, prize_location, damaged = state

    prize_x, prize_y = env.prize_positions.get(prize_location, (20, 20))  # Default to out of bounds value

    # Distance to nearest prize, 
    # (if no prize then nearest prize distance should be higher than all other possibilities)
    dist_to_prize = abs(agent_x - prize_x) + abs(agent_y - prize_y)

    # Whether the agent is damaged
    damaged_flag = 1 if damaged else 0

    # Relative position to mid point
    dist_mid_x = agent_x - 3
    dist_mid_y = agent_y - 3

    # Action-based features
    # relative movement values
    dx, dy = 0, 0
    if action == 0:  # Up
        dy = -1
    elif action == 1:  # Down
        dy = 1
    elif action == 2:  # Left
        dx = -1
    elif action == 3:  # Right
        dx = 1

    new_x, new_y = agent_x + dx, agent_y + dy

    output = np.array([
        agent_x/5.0,
        agent_y/5.0,
        prize_location,
        damaged_flag, 
        dist_to_prize/5.0,
        dist_mid_x/5.0,
        dist_mid_y/5.0,
        new_x/5.0,
        new_y/5.0,
        action
    ])
    return output

def train_approx_agent(env, agent, episodes=1000, max_steps=100):
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        state = tuple(state)
        cumulative_reward = 0

        for _ in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)

            agent.update(state, action, reward, next_state)

            state = next_state
            cumulative_reward += reward

            if done:
                break

        rewards.append(cumulative_reward)
        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Cumulative Reward: {cumulative_reward}, Epsilon: {agent.epsilon:.3f}")

    return rewards


