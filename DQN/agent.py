import numpy as np
import tensorflow as tf
from tqdm import tqdm

import os 
from logger import *

from dqn import *
from environment import *
from replay_memory import *

class Agent(object):
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess
        self.dqn = DQN(self.args, self.sess, self.memory, self.env)

        self.saver = tf.train.Saver()
        self.logger = Logger(os.path.join(self.args.dqn_log_dir, self.model_dir))

        print('Trainalbe_variables of DKVMNAgent')
        for i in tf.trainable_variables():
            if "dkvmn" not in i.op.name:
                print(i.op.name)
                #self.sess.run(tf.initialize_variables([i]))

        self.sess.run(tf.global_variables_initializer())

        self.dqn.update_target_network()


    def train(self):
        print('Agent is training')
        self.episode_count = 0
        best_reward = 0
        self.episode_reward = 0
        episode_rewards = []

        print('===== Start to make random memory =====')
        self.reset_episode()
        for self.step in tqdm(range(1, self.args.max_step+1), ncols=70, initial=0):
            action = self.select_action()

            next_state, reward, terminal = self.env.act(action)
            self.memory.add(action, reward, terminal, next_state)
            
            self.episode_reward += reward 
            if terminal:
                self.episode_count += 1
                episode_rewards.append(self.episode_reward)
                if self.episode_reward > best_reward:
                    best_reward = self.episode_reward
                self.logger.log_scalar(tag='reward', value=self.episode_reward, step=self.step)
                self.reset_episode()

            if self.step >= self.args.training_start_step:
                if self.step == self.args.training_start_step:
                    print("===== Start to update the network =====")

                if self.step % self.args.train_interval == 0:
                    loss, _ = self.dqn.train_network()

                if self.step % self.args.copy_interval == 0:
                    self.dqn.update_target_network()

                if self.step % self.args.save_interval == 0:
                    self.save()

                if self.step % self.args.show_interval == 0:
                    avg_r = np.mean(episode_rewards)
                    max_r = np.max(episode_rewards)
                    min_r = np.min(episode_rewards)
                    if max_r > best_reward:
                        best_reward = max_r
                    print('\n[recent %d episodes] avg_r: %.4f, max_r: %d, min_r: %d // Best: %d' % (len(episode_rewards), avg_r, max_r, min_r, best_reward))
                    episode_rewards = []

    def play(self, num_episode=10, load=True):
        if load:
            if not self.load():
                exit()
        best_reward = 0
        for episode in range(num_episode):
            self.reset_episode()
            current_reward = 0

            terminal = False
            while not terminal:
                action = self.select_action()
                next_state, reward, terminal = self.env.act(action)
                self.process_state(next_state)

                current_reward += reward
                if terminal:
                    break

            if current_reward > best_reward:
                best_reward = current_reward
            print('<%d> Current reward: %d' % (episode, current_reward))
            print('='*30)
            print('Best reward : %d' % (best_reward))


    def baseline(self, num_episode=1000, load=True):
        #if load:
        #    if not self.load():
        #        exit()
        self.episode_count = 1
        self.episode_reward = 0
        best_reward = 0
        for episode in range(num_episode):
            self.reset_episode()
            current_reward = 0

            terminal = False
            while not terminal:
                action, prob = self.env.baseline_act()
                print(action, prob)
                _, reward, terminal = self.env.act(action, prob[0])
                self.episode_reward += reward
            
                if terminal:
                    self.episode_count += 1
                    break

            if current_reward > best_reward:
                best_reward = current_reward
            print('<%d> Current reward: %d' % (episode, current_reward))
            print('='*30)
            print('Best reward : %d' % (best_reward))

    def select_action(self):
        if self.args.dqn_train:
            self.eps = np.max([self.args.eps_min, self.args.eps_init - (self.args.eps_init - self.args.eps_min)*(float(self.step)/float(self.args.max_exploration_step))])
        elif self.args.dqn_test:
            self.eps = self.args.eps_test

        print('Eps : %.3f' % self.eps)

        if np.random.rand() < self.eps:
            action = self.env.random_action()
            print('\nRandom action %d' % action)
        else:
            q = self.dqn.predict_Q_value(np.squeeze(self.env.value_matrix))[0]
            action = np.argmax(q)
            print('\nQ value %s and action %d' % (q,action))
        return action 

    def write_log(self, episode_count, episode_reward, case=None):
        logdir = './' + case + '_train.csv'
        if not os.path.exists(logdir):
            train_log = open(logdir, 'w')
            train_log.write('episode\t, total reward\n')
        else:
            train_log = open(logdir, 'a')
            train_log.write(str(episode_count) + '\t' + str(episode_reward) +'\n')
        
    @property
    def model_dir(self):
        return '{}_{}batch'.format(self.args.env_name, self.args.batch_size_dqn)
            
            
    def save(self):
        checkpoint_dir = os.path.join(self.args.dqn_checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, str(self.step)))
        print('*** Save at %d steps' % self.step)

    def load(self):
        print('Loading checkpoint ...')
        checkpoint_dir = os.path.join(self.args.dqn_checkpoint_dir, self.model_dir)
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_model = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
            print('Success to laod %s' % checkpoint_model)
            return True
        else:
            print('Failed to find a checkpoint')
            return False

class SimpleAgent(Agent):
    def __init__(self, args, sess):
        self.env = SimpleEnvironment(args)
        self.memory = SimpleMemory(args, self.env.state_shape)
        super(SimpleAgent, self).__init__(args, sess)

    def reset_episode(self):
        self.state = self.env.new_episode()

    def process_state(self, next_state):
        self.state = next_state


class DKVMNAgent(Agent):
    def __init__(self, args, sess, dkvmn):
        print('Initializing AGENT')

        self.env = DKVMNEnvironment(args, sess, dkvmn)
        self.memory = DKVMNMemory(args, self.env.state_shape)
        super(DKVMNAgent, self).__init__(args, sess)
        dkvmn.load()

    def reset_episode(self):
        self.env.new_episode()
        self.write_log(self.episode_count, self.episode_reward, case=self.args.case)
        self.env.episode_step = 0
        print('Episode rewards :%3.4f' % self.episode_reward)
        self.episode_reward = 0

