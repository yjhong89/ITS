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

        self.env.print_info()
        self.state_shape = self.env.get_value_memory_shape()
        self.num_actions = self.env.get_n_questions()
        self.initial_ckpt = np.copy(self.env.memory.memory_value)
        self.episode_step = 0
            

    def new_episode(self):
        self.env.load()
        #print(self.env.get_value_memory())

        #dummy_q = np.random.randint(0,self.args.n_questions, (1,1))
        #dummy_qa = np.random.randint(0,self.args.n_questions, (1,1))
        #dummy_q = np.expand_dims([1], axis=0)
        #dummy_qa = np.expand_dims([1], axis=0) 

        #value_memory_sum = tf.reduce_sum(self.env.get_value_memory())
        #print("Sum of value_memory")
        #print(self.sess.run(value_memory_sum, feed_dict={self.env.q_data_seq:dummy_q, self.env.qa_data_seq:dummy_qa}))

        #self.env.memory.memory_value = self.initial_ckpt

    def act(self, action):
        #print('\nact is not implemented\n')

        action = np.asarray(action, dtype=np.int32)
        action = np.expand_dims(np.expand_dims(action, axis=0), axis=0)

        qa, _ = self.env.update_value_memory_with_sampling_a_given_q(action)
        ## only used for network activation
        dummy_qa = np.random.randint(0,self.args.n_questions, (1,1))
        qa = self.sess.run(qa, feed_dict={self.env.q_data_seq:action, self.env.qa_data_seq:dummy_qa})


        # Need to feed value to self.env.qa_data_seq
        self.reward, self.next_state = self.sess.run([self.env.value_memory_difference, self.env.next_state], feed_dict={self.env.q_data_seq:action, self.env.qa_data_seq:qa})
        self.episode_step += 1

        print('QA %d %f' % (qa, np.sum(self.next_state)))
        if self.episode_step == self.args.episode_maxstep:
            terminal = True
        else:
            terminal = False
        # self.next_state has shape of [?,?,?,?]
        return np.squeeze(self.next_state,0), self.reward, terminal

    def random_action(self):
        return random.randrange(1, self.num_actions+1)
        
