##### Intelligent Tutoring System #####
import tensorflow as tf

import argparse, os, sys
sys.path.append('DKVMN')
sys.path.append('DQN')

from agent import *
from model import *
from data_loader import *

def str2bool(s):
    if s.lower() in ('yes', 'y', '1', 'true', 't'):
        return True
    elif s.lower() in ('no', 'n', '0', 'false', 'f'):
        return False

def setHyperParamsForDataset(args):
    if args.dataset == 'assist2009_updated':
        args.batch_size = 32
        args.memory_size = 20
        args.memory_key_state_dim = 50
        args.memory_value_state_dim = 200
        args.final_fc_dim = 50
        args.n_questions = 110
        args.seq_len = 200

    elif args.dataset == 'synthetic':
        args.batch_size = 32
        args.memory_size = 5 
        args.memory_key_state_dim = 10
        args.memory_value_state_dim = 10
        args.final_fc_dim = 50
        args.n_questions = 50
        args.seq_len = 50

    elif args.dataset == 'assist2015':
        args.batch_size = 50 
        args.memory_size = 20
        args.memory_key_state_dim = 50
        args.memory_value_state_dim = 100
        args.final_fc_dim = 50
        args.n_questions = 100
        args.seq_len = 200

def main():
    try:
        parser = argparse.ArgumentParser()

        ########## Control flag ##########
        parser.add_argument('--dkvmn_train', type=str2bool, default='f')
        parser.add_argument('--dkvmn_test', type=str2bool, default='f')
        parser.add_argument('--dqn_train', type=str2bool, default='f')
        parser.add_argument('--dqn_test', type=str2bool, default='f')
        
        ########## DKVMN ##########
        parser.add_argument('--dataset', type=str, choices=['synthetic', 'assist2009_updated','assist2015','STATICS'], default='STATICS')
        parser.add_argument('--num_epochs', type=int, default=300)
        parser.add_argument('--init_from', type=str2bool, default='t')
        parser.add_argument('--show', type=str2bool, default='f')

        parser.add_argument('--anneal_interval', type=int, default=20)
        parser.add_argument('--maxgradnorm', type=float, default=50.0)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--initial_lr', type=float, default=0.05)

        parser.add_argument('--dkvmn_checkpoint_dir', type=str, default='DKVMN/checkpoint')
        parser.add_argument('--dkvmn_log_dir', type=str, default='DKVMN/log')
        parser.add_argument('--data_dir', type=str, default='DKVMN/data')


        ##### Default(STATICS) hyperparameter #####
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument('--memory_size', type=int, default=50)
        parser.add_argument('--memory_key_state_dim', type=int, default=50)
        parser.add_argument('--memory_value_state_dim', type=int, default=100)
        parser.add_argument('--final_fc_dim', type=int, default=50)
        parser.add_argument('--n_questions', type=int, default=1223)
        parser.add_argument('--seq_len', type=int, default=200)


        ########## DQN ##########
        parser.add_argument('--env_name', type=str, choices=['CartPole-v0', 'DKVMN'], default='DKVMN')
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

        parser.add_argument('--dqn_checkpoint_dir', type=str, default='DQN/checkpoint')
        parser.add_argument('--dqn_log_dir', type=str, default='DQN/log')

        myArgs = parser.parse_args()
        setHyperParamsForDataset(myArgs)

        ### check dkvmn dir ###
        if not os.path.exists(myArgs.dkvmn_checkpoint_dir):
            os.makedirs(myArgs.dkvmn_checkpoint_dir)
        if not os.path.exists(myArgs.dkvmn_log_dir):
            os.makedirs(myArgs.dkvmn_log_dir)

        data = DATA_LOADER(myArgs.n_questions, myArgs.seq_len, ',')
        data_directory = os.path.join(myArgs.data_dir, myArgs.dataset)

        ### check dqn dir ###
        if not os.path.exists(myArgs.dqn_checkpoint_dir):
            os.makedirs(myArgs.dqn_checkpoint_dir)
        if not os.path.exists(myArgs.dqn_log_dir):
            os.makedirs(myArgs.dqn_log_dir)

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=run_config) as sess:
            
            ## TODO : rename Model to DKVMNModel
            dkvmn = Model(myArgs, sess, name='DKVMN')
            ##### DKVMN #####
            if myArgs.dkvmn_train:
                train_data_path = os.path.join(data_directory, myArgs.dataset + '_train1.csv')
                valid_data_path = os.path.join(data_directory, myArgs.dataset + '_valid1.csv')

                train_q_data, train_qa_data = data.load_data(train_data_path)
                print('Train data loaded')
                valid_q_data, valid_qa_data = data.load_data(valid_data_path)
                print('Valid data loaded')
                print('Shape of train data : %s, valid data : %s' % (train_q_data.shape, valid_q_data.shape))
                print('Start training')
                dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data)

            if myArgs.dkvmn_test:
                test_data_path = os.path.join(data_directory, args.dataset + '_test.csv')
                test_q_data, test_qa_data = data.load_data(test_data_path)
                print('Test data loaded')
                dkvmn.test(test_q_data, test_qa_data)
            
            ##### DQN #####
            if myArgs.env_name == 'CartPole-v0':
                myAgent = SimpleAgent(myArgs, sess)
            elif myArgs.env_name == 'DKVMN':
                myAgent = DKVMNAgent(myArgs, sess, dkvmn)

            if myArgs.dqn_train:
                myAgent.train()
            if myArgs.dqn_test:
                myAgent.play()
    
    except KeyboardInterrupt:
        #myArgs.display
        myArgs.train = False
        myAgent.play(3, False)
        myAgent.save()
        sess.close()

if __name__ == '__main__':
    main()
