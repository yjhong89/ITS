import numpy as np
import os, time
import tensorflow as tf
import operations
import shutil
from memory import DKVMN
from sklearn import metrics



class DKVMNModel():
    def __init__(self, args, sess, name='KT'):

        self.args = args
        self.name = name
        self.sess = sess
        #self.batch_size = args.batch_size

        tf.set_random_seed(224)

        self.init_model()
        self.init_total_prediction_probability()
    
    def sampling_a_given_q(self, q, value_matrix):
        q_embed = self.embedding_q(q)
        correlation_weight = self.memory.attention(q_embed)

        _, _, _, pred_prob = self.inference(q_embed, correlation_weight, value_matrix, reuse_flag = True)
        #pred_prob[tf.where(tf.less(pred_prob, 0.3))].assign(0.3)
        #idx = tf.gather_nd(pred_prob, tf.where(tf.less(pred_prob, 0.3)))
        #pred_prob = tf.assign(idx,tf.constant(0.3))
        pred_prob = tf.clip_by_value(pred_prob, 0.3, 1.0)

        #pred_prob = tf.minimum(pred_prob, 0.3)
        threshold = tf.random_uniform(pred_prob.shape)

        a = tf.cast(tf.less(threshold, pred_prob), tf.int32)
        qa = q + tf.multiply(a, self.args.n_questions)[0]

        return qa 

    def inference(self, q_embed, correlation_weight, value_matrix, reuse_flag):
        read_content = self.memory.value.read(value_matrix, correlation_weight)

        ##### ADD new FC layer for q_embedding. There is an layer in MXnet implementation
        q_embed_content_logit = operations.linear(q_embed, 50, name='input_embed_content', reuse=reuse_flag)
        q_embed_content = tf.tanh(q_embed_content_logit)

        mastery_level_prior_difficulty = tf.concat([read_content, q_embed_content], 1)
        #mastery_level_prior_difficulty = tf.concat([read_content, q_embed], 1)

        # f_t
        summary_logit = operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
        if self.args.summary_activation == 'tanh':
            summary_vector = tf.tanh(summary_logit)
        elif self.args.summary_activation == 'sigmoid':
            summary_vector = tf.sigmoid(summary_logit)
        elif self.args.summary_activation == 'relu':
            summary_vector = tf.nn.relu(summary_logit)

        #summary_vector = tf.sigmoid(operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag))
        #summary_vector = tf.tanh(operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag))
        # p_t
        pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag)

        pred_prob = tf.sigmoid(pred_logits)

        return read_content, summary_vector, pred_logits, pred_prob

    def init_total_prediction_probability(self):
        self.total_q_data = tf.placeholder(tf.int32, [self.args.n_questions], name='total_q_data') 
        self.total_value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='total_value_matrix')

        total_q_data = tf.constant(np.arange(1,self.args.n_questions+1))
        q_embeds = self.embedding_q(total_q_data)
        correlation_weight = self.memory.attention(q_embeds)
       
        stacked_total_value_matrix = tf.tile(tf.expand_dims(self.total_value_matrix, 0), tf.stack([self.args.n_questions, 1, 1]))
        _, _, _, self.total_pred_probs = self.inference(q_embeds, correlation_weight, stacked_total_value_matrix, True)

        
    def init_memory(self):
        with tf.variable_scope('Memory'):
            init_memory_key = tf.get_variable('key', [self.args.memory_size, self.args.memory_key_state_dim], \
                initializer=tf.random_normal_initializer(stddev=0.1))
                #initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.init_memory_value = tf.get_variable('value', [self.args.memory_size,self.args.memory_value_state_dim], \
                initializer=tf.random_normal_initializer(stddev=0.1))
                #initializer=tf.truncated_normal_initializer(stddev=0.1))
                #initializer=tf.random_uniform_initializer(minval=0.5, maxval=1.0))
                
        # Broadcast memory value tensor to match [batch size, memory size, memory state dim]
        # First expand dim at axis 0 so that makes 'batch size' axis and tile it along 'batch size' axis
        # tf.tile(inputs, multiples) : multiples length must be thes saame as the number of dimensions in input
        # tf.stack takes a list and convert each element to a tensor
        stacked_init_memory_value = tf.tile(tf.expand_dims(self.init_memory_value, 0), tf.stack([self.args.batch_size, 1, 1]))
                
        return DKVMN(self.args.memory_size, self.args.memory_key_state_dim, \
                self.args.memory_value_state_dim, init_memory_key=init_memory_key, init_memory_value=stacked_init_memory_value, args=self.args, name='DKVMN')

    def init_embedding_mtx(self):
        # Embedding to [batch size, seq_len, memory_state_dim(d_k or d_v)]
        with tf.variable_scope('Embedding'):
            # A
            self.q_embed_mtx = tf.get_variable('q_embed', [self.args.n_questions+1, self.args.memory_key_state_dim],\
                initializer=tf.random_normal_initializer(stddev=0.1))
                #initializer=tf.truncated_normal_initializer(stddev=0.1))
            # B
            self.qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.random_normal_initializer(stddev=0.1))        
            #self.qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))        
        

    def embedding_q(self, q):
        return tf.nn.embedding_lookup(self.q_embed_mtx, q)

    def embedding_qa(self, qa):
        return tf.nn.embedding_lookup(self.qa_embed_mtx, qa)
        

    def calculate_knowledge_growth(self, value_matrix, correlation_weight, qa_embed, read_content, summary, pred_prob):
        if self.args.knowledge_growth == 'origin': 
             return qa_embed
        
        elif self.args.knowledge_growth == 'value_matrix':
             value_matrix_reshaped = tf.reshape(value_matrix, [self.args.batch_size, -1])
             return tf.concat([value_matrix_reshaped, qa_embed], 1)

        elif self.args.knowledge_growth == 'read_content':
             read_content_reshaped = tf.reshape(read_content, [self.args.batch_size, -1])
             return tf.concat([read_content_reshaped, qa_embed], 1)

        elif self.args.knowledge_growth == 'summary':
             summary_reshaped = tf.reshape(summary, [self.args.batch_size, -1])
             return tf.concat([summary_reshaped, qa_embed], 1)
 
        elif self.args.knowledge_growth == 'pred_prob':
             pred_prob_reshaped = tf.reshape(pred_prob, [self.args.batch_size, -1])
             return tf.concat([pred_prob_reshaped, qa_embed], 1)

    def extract_a_from_qa(self, qa):
        return tf.cast(tf.greater(qa, tf.constant(self.args.n_questions)), tf.float32)

    def init_model(self):
        # 'seq_len' means question sequences
        self.q_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='q_data_seq') 
        self.qa_data_seq = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='qa_data')
        self.target_seq = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len], name='target')

        self.memory = self.init_memory()
        self.init_embedding_mtx()
            
        slice_q_data = tf.split(self.q_data_seq, self.args.seq_len, 1) 
        slice_qa_data = tf.split(self.qa_data_seq, self.args.seq_len, 1) 

        
        prediction = list()
        reuse_flag = False

        #self.prev_value_memory = self.memory.memory_value
        
        # Logics
        for i in range(self.args.seq_len):
            # To reuse linear vectors
            if i != 0:
                reuse_flag = True

            q = tf.squeeze(slice_q_data[i], 1)
            qa = tf.squeeze(slice_qa_data[i], 1)
            a = self.extract_a_from_qa(qa)

            q_embed = self.embedding_q(q)
            qa_embed = self.embedding_qa(qa)

            correlation_weight = self.memory.attention(q_embed)
                
            prev_read_content, prev_summary, prev_pred_logit, prev_pred_prob = self.inference(q_embed, correlation_weight, self.memory.memory_value, reuse_flag)
            prediction.append(prev_pred_logit)

            knowledge_growth = self.calculate_knowledge_growth(self.memory.memory_value, correlation_weight, qa_embed, prev_read_content, prev_summary, prev_pred_prob)
            self.memory.memory_value = self.memory.value.write_given_a(self.memory.memory_value, correlation_weight, knowledge_growth, a, reuse_flag)
            #self.memory.memory_value = self.memory.value.write(self.memory.memory_value, correlation_weight, qa_embed, knowledge_growth, reuse_flag)

        # reward 
        #self.value_memory_difference = tf.reduce_sum(self.memory.memory_value - self.prev_value_memory)
        #self.next_state = self.memory.memory_value        

        # 'prediction' : seq_len length list of [batch size ,1], make it [batch size, seq_len] tensor
        # tf.stack convert to [batch size, seq_len, 1]
        pred_logits = tf.reshape(tf.stack(prediction, axis=1), [self.args.batch_size, self.args.seq_len]) 

        # Define loss : standard cross entropy loss, need to ignore '-1' label example
        # Make target/label 1-d array
        target_1d = tf.reshape(self.target_seq, [-1])
        pred_logits_1d = tf.reshape(pred_logits, [-1])
        index = tf.where(tf.not_equal(target_1d, tf.constant(-1., dtype=tf.float32)))
        # tf.gather(params, indices) : Gather slices from params according to indices
        filtered_target = tf.gather(target_1d, index)
        filtered_logits = tf.gather(pred_logits_1d, index)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
        #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
        self.pred = tf.sigmoid(pred_logits)

        # Optimizer : SGD + MOMENTUM with learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        #self.lr_decay = tf.train.exponential_decay(self.args.initial_lr, global_step=global_step, decay_steps=10000, decay_rate=0.667, staircase=True)
        self.learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, decay_steps=self.args.anneal_interval*(tf.shape(self.q_data_seq)[0] // self.args.batch_size), decay_rate=0.667, staircase=True)
#        self.learning_rate = tf.maximum(lr, self.args.lr_lowerbound)
        optimizer = tf.train.MomentumOptimizer(self.lr, self.args.momentum)
        #optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum)
        grads, vrbs = zip(*optimizer.compute_gradients(self.loss))
        ## grad, _ = tf.clip_by_global_norm(grads, self.args.maxgradnorm)
        self.grads = grads
        #print('\nGrad')
        #print(len(grads))
        #for i in range(len(grads)):
            #print(tf.shape(grads[i][0]))
        #self.global_norm = tf.global_norm(grads)
        grad, self.global_norm = tf.clip_by_global_norm(grads, self.args.maxgradnorm)
        #grad, _ = tf.clip_by_global_norm(grads, self.args.maxgradnorm, use_norm = self.global_norm)
        
        self.train_op = optimizer.apply_gradients(list(zip(grad, vrbs)), global_step=self.global_step)
#        grad_clip = [(tf.clip_by_value(grad, -self.args.maxgradnorm, self.args.maxgradnorm), var) for grad, var in grads]
        self.tr_vrbs = tf.trainable_variables()
        for i in self.tr_vrbs:
            print(i.name)
            print(i.shape)

        self.saver = tf.train.Saver()
        print('Finish init_model')


    def train(self, train_q_data, train_qa_data, valid_q_data, valid_qa_data):
        #np.random.seed(224)
        # q_data, qa_data : [samples, seq_len]

        training_step = train_q_data.shape[0] // self.args.batch_size
        self.sess.run(tf.global_variables_initializer())
        '''
        value_mem = self.init_memory_value.eval()
        print(np.sum(value_mem))

        for i in self.tr_vrbs:
            print(i.name)
            print(i.shape)
            print(np.sum(i.eval()))
        '''
        
        if self.args.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=training_step)

        self.train_count = 0
        if self.args.init_from:
            if self.load():
                print('Checkpoint_loaded')
            else:
                print('No checkpoint')
        else:
            if os.path.exists(os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)):
                try:
                    shutil.rmtree(os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir))
                    shutil.rmtree(os.path.join(self.args.dkvmn_log_dir, self.model_dir+'.csv'))
                except(FileNotFoundError, IOError) as e:
                    print('[Delete Error] %s - %s' % (e.filename, e.strerror))
        
        best_valid_auc = 0
        #print(self.args.seq_len)

        # Training
        for epoch in range(0, self.args.num_epochs):
            shuffle_index = np.random.permutation(train_q_data.shape[0])
            q_data_shuffled = train_q_data[shuffle_index, :]
            qa_data_shuffled = train_qa_data[shuffle_index, :]

            if self.args.show:
                bar.next()

            pred_list = list()
            target_list = list()        
            epoch_loss = 0
            learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, decay_steps=self.args.anneal_interval*training_step, decay_rate=0.667, staircase=True)
            lr = learning_rate.eval()
            #print('LR %f' % lr )
            #print('Epoch %d starts with learning rate : %3.5f' % (epoch+1, self.sess.run(learning_rate)))
            for steps in range(training_step):
                # [batch size, seq_len]
                q_batch_seq = q_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                qa_batch_seq = qa_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
    
                # qa : exercise index + answer(0 or 1)*exercies_number
                # right : 1, wrong : 0, padding : -1
                target = qa_batch_seq[:,:]
                # Make integer type to calculate target
                target = target.astype(np.int)
                target_batch = (target - 1) // self.args.n_questions  
                target_batch = target_batch.astype(np.float)

                feed_dict = {self.q_data_seq:q_batch_seq, self.qa_data_seq:qa_batch_seq, self.target_seq:target_batch, self.lr:self.args.initial_lr}
                #self.lr:self.sess.run(learning_rate)
                #loss_, pred_, _, = self.sess.run([self.loss, self.pred, self.train_op], feed_dict=feed_dict)
                loss_, pred_, _, global_norm, grads, _lr = self.sess.run([self.loss, self.pred, self.train_op, self.global_norm, self.grads, self.learning_rate], feed_dict=feed_dict)
                #print('Global norm %f' % global_norm)
                #print(grads)
                #print('Legnth of global variables : %d' % len(grads))

                #print('LR %f %f' % (lr, _lr))
                #print(len(grads[0])) # 20
                #print(len(grads[0][0])) # 50

                #print(np.squre(grads[0]))
                '''
                globbal_norm = 0
                for i in range(len(grads)):
                    print(i)
                    print(np.sum(np.square(grads[i])))
                    global_norm += np.sum(np.square(grads[i]))
                print(np.sqrt(global_norm))
                '''
                # Get right answer index
                # Make [batch size * seq_len, 1]
                right_target = np.asarray(target_batch).reshape(-1,1)
                right_pred = np.asarray(pred_).reshape(-1,1)
                # np.flatnonzero returns indices which is nonzero, convert it list 
                right_index = np.flatnonzero(right_target != -1.).tolist()
                #print(len(right_index)/self.args.batch_size)
                # Number of 'training_step' elements list with [batch size * seq_len, ]
                pred_list.append(right_pred[right_index])
                target_list.append(right_target[right_index])

                epoch_loss += loss_
                #print('Epoch %d/%d, steps %d/%d, loss : %3.5f' % (epoch+1, self.args.num_epochs, steps+1, training_step, loss_))
                

            if self.args.show:
                bar.finish()        
            
            all_pred = np.concatenate(pred_list, axis=0)
            all_target = np.concatenate(target_list, axis=0)

            # Compute metrics
            self.auc = metrics.roc_auc_score(all_target, all_pred)
            # Extract elements with boolean index
            # Make '1' for elements higher than 0.5
            # Make '0' for elements lower than 0.5
            all_pred[all_pred > 0.5] = 1.0
            all_pred[all_pred <= 0.5] = 0.0
            self.accuracy = metrics.accuracy_score(all_target, all_pred)

            epoch_loss = epoch_loss / training_step    
            print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (epoch+1, self.args.num_epochs, epoch_loss, self.auc, self.accuracy))
            self.write_log(epoch=epoch+1, auc=self.auc, accuracy=self.accuracy, loss=epoch_loss, name='training_')

            valid_steps = valid_q_data.shape[0] // self.args.batch_size
            valid_pred_list = list()
            valid_target_list = list()
            for s in range(valid_steps):
                # Validation
                valid_q = valid_q_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                valid_qa = valid_qa_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                # right : 1, wrong : 0, padding : -1
                valid_target = (valid_qa - 1) // self.args.n_questions
                valid_feed_dict = {self.q_data_seq : valid_q, self.qa_data_seq : valid_qa, self.target_seq : valid_target}
                valid_loss, valid_pred = self.sess.run([self.loss, self.pred], feed_dict=valid_feed_dict)
                # Same with training set
                valid_right_target = np.asarray(valid_target).reshape(-1,)
                valid_right_pred = np.asarray(valid_pred).reshape(-1,)
                valid_right_index = np.flatnonzero(valid_right_target != -1).tolist()    
                valid_target_list.append(valid_right_target[valid_right_index])
                valid_pred_list.append(valid_right_pred[valid_right_index])
            
            all_valid_pred = np.concatenate(valid_pred_list, axis=0)
            all_valid_target = np.concatenate(valid_target_list, axis=0)

            valid_auc = metrics.roc_auc_score(all_valid_target, all_valid_pred)
             # For validation accuracy
            all_valid_pred[all_valid_pred > 0.5] = 1.0
            all_valid_pred[all_valid_pred <= 0.5] = 0.0
            valid_accuracy = metrics.accuracy_score(all_valid_target, all_valid_pred)
            print('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' %(epoch+1, self.args.num_epochs, valid_auc, valid_accuracy))
            # Valid log
            self.write_log(epoch=epoch+1, auc=valid_auc, accuracy=valid_accuracy, loss=valid_loss, name='valid_')
            if valid_auc > best_valid_auc:
                print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
                best_valid_auc = valid_auc
                best_epoch = epoch + 1
                self.save(best_epoch)

        return best_epoch    
    
        
             
    
    def test(self, test_q, test_qa):
        steps = test_q.shape[0] // self.args.batch_size
        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

        print('Initial value of probability')
        print(init_probability)

        pred_list = list()
        target_list = list()

        for s in range(steps):
            test_q_batch = test_q[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            test_qa_batch = test_qa[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            target = test_qa_batch[:,:]
            target = target.astype(np.int)
            target_batch = (target - 1) // self.args.n_questions  
            target_batch = target_batch.astype(np.float)
            feed_dict = {self.q_data_seq:test_q_batch, self.qa_data_seq:test_qa_batch, self.target_seq:target_batch}
            loss_, pred_ = self.sess.run([self.loss, self.pred], feed_dict=feed_dict)
            # Get right answer index
            # Make [batch size * seq_len, 1]
            right_target = np.asarray(target_batch).reshape(-1,1)
            right_pred = np.asarray(pred_).reshape(-1,1)
            # np.flatnonzero returns indices which is nonzero, convert it list 
            right_index = np.flatnonzero(right_target != -1.).tolist()
            # Number of 'training_step' elements list with [batch size * seq_len, ]
            pred_list.append(right_pred[right_index])
            target_list.append(right_target[right_index])

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)

        test_auc = metrics.roc_auc_score(all_target, all_pred)
        # Compute metrics
        all_pred[all_pred > 0.5] = 1.0
        all_pred[all_pred <= 0.5] = 0.0
        # Extract elements with boolean index
        # Make '1' for elements higher than 0.5
        # Make '0' for elements lower than 0.5

        test_accuracy = metrics.accuracy_score(all_target, all_pred)

        print('Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy))
        self.write_log(epoch=1, auc=test_auc, accuracy=test_accuracy, loss=0, name='test_')

        log_file_name = 'test_all_%s.txt' % self.args.gpu_id 
        log_file = open(log_file_name, 'a')
        log = 'Test auc : %3.4f, Test accuracy : %3.4f' % (test_auc, test_accuracy)
        log_file.write(self.model_dir + '\n')
        log_file.write(log + '\n') 
        log_file.flush()    
        

########################################################## FOR Reinforcement Learning ##############################################################

    def init_step(self):
        # q : action for RL
        # value_matrix : state for RL
        self.q = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='step_q') 
        self.a = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='step_a') 
        self.value_matrix = tf.placeholder(tf.float32, [self.args.memory_size,self.args.memory_value_state_dim], name='step_value_matrix')

        slice_a = tf.split(self.a, self.args.seq_len, 1) 
        a = tf.squeeze(slice_a[0], 1)
 
        slice_q = tf.split(self.q, self.args.seq_len, 1) 
        q = tf.squeeze(slice_q[0], 1)
        q_embed = self.embedding_q(q)
        correlation_weight = self.memory.attention(q_embed)

        stacked_value_matrix = tf.tile(tf.expand_dims(self.value_matrix, 0), tf.stack([self.args.batch_size, 1, 1]))
         
        # -1 for sampling
        # 0, 1 for given answer
        self.qa = tf.cond(tf.squeeze(a) < 0, lambda: self.sampling_a_given_q(q, stacked_value_matrix), lambda: q + tf.multiply(a, self.args.n_questions))
        a = (self.qa-1) // self.args.n_questions
        qa_embed = self.embedding_qa(self.qa) 

        ######### Before Step ##########
        prev_read_content, prev_summary, prev_pred_logits, prev_pred_prob = self.inference(q_embed, correlation_weight, stacked_value_matrix, reuse_flag = True)

        ######### STEP #####################
        knowledge_growth = self.calculate_knowledge_growth(stacked_value_matrix, correlation_weight, qa_embed, prev_read_content, prev_summary, prev_pred_prob)
        # TODO : refactor sampling_a_given_q to return a only for below function call
        self.stepped_value_matrix = tf.squeeze(self.memory.value.write_given_a(stacked_value_matrix, correlation_weight, knowledge_growth, a, True), axis=0)
        #self.stepped_value_matrix = tf.squeeze(self.memory.value.write(stacked_value_matrix, correlation_weight, qa_embed, knowledge_growth, True), axis=0)
        self.stepped_read_content, self.stepped_summary, self.stepped_pred_logits, self.stepped_pred_prob = self.inference(q_embed, correlation_weight, self.stepped_value_matrix, reuse_flag = True)

        ######### After Step #########
        self.value_matrix_difference = tf.squeeze(tf.reduce_sum(self.stepped_value_matrix - stacked_value_matrix))
        self.read_content_difference = tf.squeeze(tf.reduce_sum(self.stepped_read_content - prev_read_content))
        self.summary_difference = tf.squeeze(tf.reduce_sum(self.stepped_summary - prev_summary))
        self.pred_logit_difference = tf.squeeze(tf.reduce_sum(self.stepped_pred_logits - prev_pred_logits))
        self.pred_prob_difference = tf.squeeze(tf.reduce_sum(self.stepped_pred_prob - prev_pred_prob))

    def ideal_test(self):
        type_list = [0, 1]
        for t in type_list: 
            self.ideal_test_given_type(t) 
    

    def ideal_test_given_type(self, input_type): 
        
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

        log_file_name = 'logs/'+self.model_dir
        if input_type == 0:
            log_file_name = log_file_name + '_neg.csv'
        elif input_type == 1:
            log_file_name = log_file_name + '_pos.csv' 
        elif input_type == -1:
            log_file_name = log_file_name + '_rand.csv' 

        log_file = open(log_file_name, 'w')
        value_matrix = self.sess.run(self.init_memory_value)
        for i in range(100):

            for q_idx in range(1, self.args.n_questions+1):
                q = np.expand_dims(np.expand_dims(q_idx, axis=0), axis=0) 
                a = np.expand_dims(np.expand_dims(input_type, axis=0), axis=0) 
        
                ops = [self.stepped_value_matrix, self.stepped_pred_prob, self.value_matrix_difference, self.read_content_difference, self.summary_difference, self.pred_logit_difference, self.pred_prob_difference]
                feed_dict = { self.q : q, self.a : a, self.value_matrix: value_matrix }

                value_matrix, pred_prob, value_matrix_diff, read_content_diff, summary_diff, pred_logit_diff, pred_prob_diff = np.squeeze(self.sess.run(ops, feed_dict=feed_dict))
                pred_prob = np.squeeze(np.squeeze(pred_prob))

                log = str(i)+','+ str(q_idx) +','+str(input_type)+','+str(np.sum(value_matrix))+','+str(pred_prob) + ','
                log = log + str(value_matrix_diff) + ','  + str(read_content_diff) + ',' + str(summary_diff) + ',' + str(pred_logit_diff) + ',' + str(pred_prob_diff) + '\n'  
                log_file.write(log) 

        log_file.flush()    

    @property
    def model_dir(self):
        return '{}Knowledge_{}_Summary_{}_Add_{}_Erase_{}_WriteType_{}_{}_lr{}_{}epochs'.format(self.args.prefix, self.args.knowledge_growth, self.args.summary_activation, self.args.add_signal_activation, self.args.erase_signal_activation, self.args.write_type, self.args.dataset, self.args.initial_lr, self.args.num_epochs)

    def load(self):
        #self.args.batch_size = 32
        checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print('DKVMN ckpt loaded')
            return True
        else:
            print('DKVMN cktp not loaded')
            return False

    def save(self, global_step):
        model_name = 'DKVMN'
        checkpoint_dir = os.path.join(self.args.dkvmn_checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print('Save checkpoint at %d' % (global_step+1))

    # Log file
    def write_log(self, auc, accuracy, loss, epoch, name='training_'):
        log_path = os.path.join(self.args.dkvmn_log_dir, name+self.model_dir+'.csv')
        if not os.path.exists(log_path):
            self.log_file = open(log_path, 'w')
            self.log_file.write('Epoch\tAuc\tAccuracy\tloss\n')
        else:
            self.log_file = open(log_path, 'a')    
        
        self.log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\t' + str(loss) + '\n')
        self.log_file.flush()    
        
