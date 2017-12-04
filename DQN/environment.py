import gym
import tensorflow as tf
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

        self.sess = sess
        print('Initializing ENVIRONMENT')
        self.env = dkvmn 

        self.num_actions = self.args.n_questions
        self.initial_ckpt = np.copy(self.env.memory.memory_value)
        self.episode_step = 0

        self.starting_value_matrix = self.sess.run(self.env.init_memory_value)  
        self.value_matrix = self.starting_value_matrix
        self.state_shape = list(self.value_matrix.shape)
        print(self.value_matrix.shape)
        print(np.sum(self.value_matrix))
            

    def new_episode(self):
        print('NEW EPISODE')
        self.value_matrix = self.starting_value_matrix

    def act(self, action):
        action = np.asarray(action, dtype=np.int32)
        action = np.expand_dims(np.expand_dims(action, axis=0), axis=0)

        # -1 for sampling 
        # 0, 1 for input given
        # 0 : worst, 1 : best 
        answer = np.asarray(-1, dtype=np.int32)
        answer = np.expand_dims(np.expand_dims(answer, axis=0), axis=0)

        prev_value_matrix = self.value_matrix

        self.value_matrix, qa = self.sess.run([self.env.stepped_value_matrix, self.env.qa], feed_dict={self.env.q: action, self.env.a: answer, self.env.value_matrix: prev_value_matrix})
        self.reward = np.sum(self.value_matrix) - np.sum(prev_value_matrix)

        self.episode_step += 1

        print('QA : %3d, Reward : %f' % (qa, self.reward))
        if self.episode_step == self.args.episode_maxstep:
            terminal = True
        else:
            terminal = False

        return np.squeeze(self.value_matrix), self.reward, terminal

    def random_action(self):
        return random.randrange(2, self.num_actions+1)
