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

        self.sess = sess

        self.env = dkvmn 
        dkvmn.args.seq_len = 1
        dkvmn.args.batch_size = 1

        self.env.print_info()
        self.state_shape = self.env.get_value_memory_shape()
        print('State shape')
        print(self.state_shape)
        self.num_actions = self.env.get_n_questions()

    def new_episode(self):
        print('\nnew_episode is not implemented\n')
        return False

    def act(self, action):
        #print('\nact is not implemented\n')

        action = np.asarray(action, dtype=np.int32)
        action = np.expand_dims(np.expand_dims(action, axis=0), axis=0)

        qa, _ = self.env.update_value_memory_with_sampling_a_given_q(action)
        dummy_qa = np.random.randint(0,self.args.n_questions, (1,1))
        qa = self.sess.run(qa, feed_dict={self.env.q_data_seq:action, self.env.qa_data_seq:dummy_qa})

        #self.state = self.sess.run(self.env.updated_value_memory, self.env.value_memory_difference, feed_dict={self.env.q_data_seq:action})
        # Need to feed value to self.env.qa_data_seq
        self.reward, self.next_state = self.sess.run([self.env.value_memory_difference, self.env.next_state], feed_dict={self.env.q_data_seq:action, self.env.qa_data_seq:qa})
        # self.env.qa_data_seq:qa})
        print('REWARD : ')
        print(self.reward)
        print(self.next_state)
        return self.next_state, self.reward, False

    def random_action(self):
        return random.randrange(1, self.num_actions+1)
        
