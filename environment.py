import gym
import numpy as np

class Environment(object):
    def __init__(self, args):
        self.args = args
        self.env = gym.make(self.args.env_name)
        self.num_actions = self.env.action_space.n


    def random_action(self):
        return self.env.action_space.sample()
        
class SimpleEnvironment(Environment):
    def __init__(self, args):
        super(SimpleEnvironment, self).__init__(args)
        self.frame_shape = list(self.env.observation_space.shape)

    def new_episode(self):
        return self.env.reset()

    def act(self, action):
        self.state, self.reward, self.terminal, _ = self.env.step(action)

        return self.state, self.reward, self.terminal
