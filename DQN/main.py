import tensorflow as tf

import argparse, os, sys

from agent import *

def str2bool(s):
    if s.lower() in ('yes', 'y', '1', 'true', 't'):
        return True
    elif s.lower() in ('no', 'n', '0', 'false', 'f'):
        return False

def main():
    try:
        parser = argparse.ArgumentParser()
        #parser.add_argument('--', type=, default=)

        parser.add_argument('--env_name', type=str, default='CartPole-v0', help='CartPole-v0, DKVMN')
        #parser.add_argument('--simple', type=str2bool, default='false')

        parser.add_argument('--train', type=str2bool, default='true')
        parser.add_argument('--batch_size_dqn', type=int, default=32)
        parser.add_argument('--max_step', type=int, default=10000000)
        parser.add_argument('--max_exploration_step', type=int, default=1000000)

        parser.add_argument('--replay_memory_size', type=int, default=10000)

        parser.add_argument('--discount_factor', type=float, default=0.95)
        parser.add_argument('--eps_init', type=float, default=1.0)
        parser.add_argument('--eps_min', type=float, default=0.1)
        parser.add_argument('--eps_test', type=float, default=0.05)

        parser.add_argument('--training_start_step', type=int, default=100)
        parser.add_argument('--train_interval', type=int, default=1)
        parser.add_argument('--copy_interval', type=int, default=5000)
        parser.add_argument('--save_interval', type=int, default=50000)
        parser.add_argument('--show_interval', type=int, default=2000)

        parser.add_argument('--learning_rate', type=float, default=0.001)

        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
        parser.add_argument('--log_dir', type=str, default='./log')
        #parser.add_argument('--', type=, default=)
        #parser.add_argument('--', type=, default=)
        #parser.add_argument('--', type=, default=)

        myArgs = parser.parse_args()
        if not os.path.exists(myArgs.checkpoint_dir):
            os.makedirs(myArgs.checkpoint_dir)
        if not os.path.exists(myArgs.log_dir):
            os.makedirs(myArgs.log_dir)

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=run_config) as sess:
            if myArgs.env_name == 'CartPole-v0':
                myAgent = SimpleAgent(myArgs, sess)
            elif myArgs.env_name == 'DKVMN':
                myAgent = DKVMNAgent(myArgs, sess)

            if myArgs.train:
                myAgent.train()
            else:
                myAgent.play()
    
    except KeyboardInterrupt:
        #myArgs.display
        myArgs.train = False
        myAgent.play(3, False)
        myAgent.save()
        sess.close()

if __name__ == '__main__':
    main()
