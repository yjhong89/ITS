import gym
import numpy as np
import random

#from model import *

class Environment(object):
    def __init__(self, args):
        self.args = args

        
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

    def random_action(self):
        return self.env.action_space.sample()

class DKVMNEnvironment(Environment):
    def __init__(self, args, sess, dkvmn):
        super(DKVMNEnvironment, self).__init__(args)

        self.env = dkvmn 
        self.env.print_info()
        self.state_shape = self.env.get_value_memory_shape()
        print('State shape')
        print(self.state_shape)
        self.num_actions = self.env.get_n_questions()

    def new_episode(self):
        print('\nnew_episode is not implemented\n')
        return False

    def act(self, action):
        print('\nact is not implemented\n')
        ## dkvmn predict + value_memory_update
        return self.evn.k(act)



    def random_action(self):
        return random.randrange(1, self.num_actions+1)
        
