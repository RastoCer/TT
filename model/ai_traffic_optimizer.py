import numpy as np
import random
import os

class TrafficLightOptimizer:
    def __init__(self, num_phases, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        """
        Initializes the TrafficLightOptimizer with Q-learning.

        Args:
            num_phases (int): Number of traffic light phases.
            learning_rate (float): Learning rate for Q-learning updates.
            discount_factor (float): Discount factor for future rewards.
            exploration_rate (float): Initial exploration rate for epsilon-greedy policy.
            exploration_decay (float): Decay factor for exploration rate.
        """
        self.num_phases = num_phases
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        # Initialize Q-table
        self.q_table = {}

    def get_state_key(self, state):
        """
        Converts the state into a hashable key for the Q-table.

        Args:
            state (tuple): Traffic light state (e.g., traffic densities).

        Returns:
            str: Hashable key for the Q-table.
        """
        return str(state)

    def choose_action(self, state):
        """
        Chooses an action using epsilon-greedy policy.

        Args:
            state (tuple): Current traffic state.

        Returns:
            int: Chosen action (traffic light phase).
        """
        state_key = self.get_state_key(state)
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_phases - 1)  # Explore: Random action
        return np.argmax(self.q_table.get(state_key, np.zeros(self.num_phases)))  # Exploit: Best known action

    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table using the Q-learning algorithm.

        Args:
            state (tuple): Current traffic state.
            action (int): Chosen action (traffic light phase).
            reward (float): Observed reward.
            next_state (tuple): Next traffic state after taking the action.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_phases)

        max_future_q = np.max(self.q_table.get(next_state_key, np.zeros(self.num_phases)))
        current_q = self.q_table[state_key][action]

        # Q-learning formula
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state_key][action] = new_q

        # Decay exploration rate
        self.exploration_rate = max(0.1, self.exploration_rate * self.exploration_decay)

    def save_model(self, file_path):
        """
        Saves the Q-table to a file.

        Args:
            file_path (str): File path to save the model.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, self.q_table)

    def load_model(self, file_path):
        """
        Loads the Q-table from a file.

        Args:
            file_path (str): File path to load the model from.
        """
        if os.path.exists(file_path):
            self.q_table = np.load(file_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(f"Model file '{file_path}' not found.")
