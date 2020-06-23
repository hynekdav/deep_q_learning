import keras
from collections import deque
import numpy as np
import random
import gym


class DQNLearner(object):
    """
    Class implementing training for Deep Q Network. It handles the batch learning,
    prediction of next move base on epsilon-greedy policy and memory for learning.
    """
    def __init__(self, model: keras.models.Model, action_space: gym.Env.action_space):
        self.EXPLORATION_MAX = 1.0
        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_DECAY = 0.995
        self.GAMMA = 0.95
        self.exploration_rate = self.EXPLORATION_MAX

        self.BATCH_SIZE = 64
        self.MEMORY_SIZE = 100000

        self.model = model
        self.action_space = action_space
        self.memory = deque(maxlen=self.MEMORY_SIZE)

    def save_observation(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> None:
        """
        Saves given tuple of current environment state, action, reward, next_state and done.
        :param state: Current state of the world.
        :param action: Current taken action.
        :param reward: Reward for the two parameters above.
        :param next_state: State we transitioned into.
        :param done: Indicator whether the next_state is terminal or not.
        :return: None
        """
        self.memory.append((state, action, reward, next_state, done))

    def next_move(self, state: np.array) -> int:
        """
        Predicts the next move given the current state of the world.
        With changing probability decides not to predict but to explore the state space.
        :param state: Current state of the world.
        :return: int, code of the action to be taken.
        """
        if np.random.uniform() < self.exploration_rate:
            return self.action_space.sample()
        values = self.model.predict(state)
        return int(np.argmax(values[0]))

    def learn(self) -> None:
        """
        Performs training of the underlying neural network based on the randomly sampled batch of values.
        :return: None
        """
        if len(self.memory) < self.BATCH_SIZE:
            return

        batch = random.sample(self.memory, self.BATCH_SIZE)

        states = np.array([i[0] for i in batch])
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        next_states = np.array([i[3] for i in batch])
        terminals = np.array([i[4] for i in batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        q_updates = rewards + self.GAMMA * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - terminals)
        q_values = self.model.predict_on_batch(states)
        q_values[[np.array([i for i in range(self.BATCH_SIZE)])], [actions]] = q_updates

        self.model.fit(states, q_values, epochs=1, verbose=0)

        self.exploration_rate *= self.EXPLORATION_DECAY
        self.exploration_rate = np.maximum(self.EXPLORATION_MIN, self.exploration_rate)
