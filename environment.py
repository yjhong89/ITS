import gym
import numpy as np

class Environment(object):
    def __init__(self, args):
        self.args = args

    def random_action(self):
        return self.env.action_space.sample()
        
class SimpleEnvironment(Environment):
    def __init__(self, args):
        super(SimpleEnvironment, self).__init__(args)
        self.env = gym.make(self.args.env_name)
        self.num_actions = self.env.action_space.n
        self.state_shape = list(self.env.observation_space.shape)

    def new_episode(self):
        return self.env.reset()

    def act(self, action):
        self.state, self.reward, self.terminal, _ = self.env.step(action)

        return self.state, self.reward, self.terminal


class DKVMNEnvironment(Environment):
    def __init__(self, args):
        super(DKVMNEnvironment, self).__init__(args)
        '''
        self.env = 
        self.num_actions = 
        '''
    def new_episode(self):
        return False

    def act(self, action):
        return False 

