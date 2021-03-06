import numpy as np

from agents.abstract_agent import AbstractAgent
from agents.dqn.deep_q_learner import DQNLearner

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU


class DQNAgentSimple(AbstractAgent):
    """
    An agent implementing simple feedforward neural network to land the lander.
    """

    def __init__(self, env, number_of_epochs):
        super().__init__(env, number_of_epochs)

        model = Sequential()
        model.add(Dense(64, input_shape=(self.env.observation_space.shape[0],)))
        model.add(LeakyReLU())
        model.add(Dense(64))
        model.add(LeakyReLU())
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(optimizer='adam', loss='mse')

        self.learner = DQNLearner(model=model, action_space=self.env.action_space)
        self.env.seed(42)

    def train(self):
        print()
        print(f'Training {self} for {self.number_of_epochs} episodes.')
        scores = []
        for episode in range(self.number_of_epochs):
            reward, done = 0, False
            score = 0
            observation = self.env.reset()
            state = np.reshape(observation, (1, observation.shape[0]))
            t = 0
            while not done:
                t += 1
                self.env.render()
                action = self.make_step(state)
                observation, reward, done, info = self.env.step(action)
                score += reward
                next_state = np.reshape(observation, (1, observation.shape[0]))
                self.learner.save_observation(state, action, reward, next_state, done)
                self.learner.learn()
                state = next_state
                if done:
                    print(f"Episode {episode + 1} finished after {t + 1} steps with reward {score}.")
                    scores.append(score)
        self.env.close()
        return scores

    def make_step(self, observation):
        return self.learner.next_move(observation)

    def __repr__(self):
        return f"Simple DQN Agent on {self.env.unwrapped.spec.id}"
