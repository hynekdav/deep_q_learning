from abc import ABC, abstractmethod
import gym
from typing import List
import numpy as np


class AbstractAgent(ABC):
    """
    Abstract base class for agents in the work. Provides two fields: env for the environment and number_of_epochs.
    """

    def __init__(self, env: str, number_of_epochs: int):
        """
        Initializes the agent.
        :param env: The environment in which the agent plays Blackjack.
        :param number_of_epochs: Number of epochs to train on.
        """
        self.env = gym.make(env)
        self.number_of_epochs = number_of_epochs
        super().__init__()

    @abstractmethod
    def train(self) -> List[float]:
        """
        This method should train the agent by repeatedly playing the game.
        :return: List of scores over epochs.
        """
        pass

    @abstractmethod
    def make_step(self, observation: np.array) -> int:
        """
        This method decides the next step based on the observation.
        :param observation: Gym observation. Numpy array.
        :return: int, Number of action to perform.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Returns name of the agent (for saving evaluation plot).
        :return: string, name of the agent
        """
        pass
