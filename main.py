import matplotlib.pyplot as plt
import numpy as np

from agents.abstract_agent import AbstractAgent
from agents.dqn_agent_advanced import DQNAgentAdvanced
from agents.dqn_agent_medium import DQNAgentMedium
from agents.dqn_agent_simple import DQNAgentSimple
from agents.random_agent import RandomAgent


def evaluate(agent: AbstractAgent):
    plt.close()

    scores = agent.train()
    average = np.average(scores)
    print(f'{agent} average score: {average}')
    indices = list(range(len(scores)))

    p = np.polyfit(indices, scores, 1)
    p = np.poly1d(p)

    plt.plot(scores, label='score per run')
    plt.plot(p(indices), label='trend')
    plt.plot([average] * len(scores), '-.', label='average')
    plt.xlabel('runs')
    plt.title(str(agent))

    plt.legend(loc='upper left')
    plt.grid(True)

    plt.savefig(f'{agent}.png')


if __name__ == '__main__':
    params = {'CartPole-v0': 1000, 'LunarLander-v2': 250}

    for environment, episodes in params.items():
        agents = [DQNAgentSimple(environment, episodes), DQNAgentMedium(environment, episodes),
                  DQNAgentAdvanced(environment, episodes), RandomAgent(environment, episodes)]
        for agent in agents:
            evaluate(agent)
