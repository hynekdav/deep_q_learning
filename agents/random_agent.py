from agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):
    """
    Very simple random agent used as a baseline.
    """

    def __init__(self, env: str, number_of_epochs: int):
        super().__init__(env, number_of_epochs)
        self.env.seed(42)

    def train(self):
        print()
        print(f'Training {self} for {self.number_of_epochs} episodes.')
        scores = []
        for episode in range(self.number_of_epochs):
            reward, done = 0, False
            score = 0
            observation = self.env.reset()
            t = 0
            while not done:
                t += 1
                self.env.render()
                action = self.make_step(observation)
                observation, reward, done, info = self.env.step(action)
                score += reward
                if done:
                    print(f"Episode {episode + 1} finished after {t + 1} steps with reward {score}.")
                    scores.append(score)
        self.env.close()
        return scores

    def make_step(self, observation) -> int:
        return self.env.action_space.sample()

    def __repr__(self):
        return f'Random Agent on {self.env.unwrapped.spec.id}'
