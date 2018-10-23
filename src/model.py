import math
import timeit
import numpy as np
import tensorflow as tf
import multiprocessing as mp

class TransE:
    def __init__(self, kg, embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator

        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])

        self.train_op = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None

        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None

        self.skent_embeddings = tf.placeholder(tf.float32, [kg.n_entity, embedding_dim])

        bound = 6 / math.sqrt(self.embedding_dim)

        '''embeddings'''
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)

        self.entity_embedding = self.entity_embedding + self.skent_embeddings


        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)

        with tf.name_scope('training'):

            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg

        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):

        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)

            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg), name='max_margin_loss')
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, session, summary_writer,skp_file):
        skentity_embeddings = np.load(skp_file)
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        for i in range(n_batch):
            batch_tripos, batch_trineg = training_batch_queue.get()

            batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_tripos,
                                                            self.triple_neg: batch_trineg,
                                                            self.margin: [self.margin_value] * len(batch_tripos),
                                                            self.skent_embeddings: skentity_embeddings
                                                            })
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            n_used_triple += len(batch_tripos)
            print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                            n_used_triple,
                                                                            self.kg.n_training_triple,
                                                                            batch_loss / (len(batch_tripos))))
        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish training-----')
        return epoch_loss

    def launch_evaluation(self, session, test_data, skp_file, save_file):
        skentity_embeddings = np.load(skp_file)
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in test_data:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple,
                                                                              self.skent_embeddings: skentity_embeddings
                                                                              })
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
            print('[{:.3f}s] #evaluation triple: {}'.format(timeit.default_timer() - start,
                                                               n_used_eval_triple))
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_meanrank_filter = 0
        head_hits10_filter = 0
        tail_meanrank_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_meanrank_filter += tail_rank_filter
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        file = open(save_file, "a+")

        #head_meanrank_raw /= n_used_eval_triple
        #head_hits10_rawr = head_hits10_raw/n_used_eval_triple
        #tail_meanrank_raw /= n_used_eval_triple
        #tail_hits10_rawr = tail_hits10_raw/n_used_eval_triple
        file.write("-----Raw-----\n")
        file.write("-----Head prediction-----\n")
        file.write("head_meanrank_raw \t head_hits10_raw@10\n")
        file.write(str(head_meanrank_raw))
        file.write("\t")
        file.write(str(head_hits10_raw))
        file.write("\n")
        file.write("-----Tail prediction-----\n")
        file.write("tail_meanrank_raw \t tail_hits10_raw tailHits \n")
        file.write(str(tail_meanrank_raw))
        file.write("\t")
        file.write(str(tail_hits10_raw))
        file.write("\n")

        print(n_used_eval_triple)
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}'.format(head_meanrank_raw))
        print('Hits@10: {:.3f}'.format(head_hits10_raw))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}'.format(tail_meanrank_raw))
        print('Hits@10: {:.3f}'.format(tail_hits10_raw))

        print('-----Filter-----')
        #head_meanrank_filter /= n_used_eval_triple
        #head_hits10_filterf = head_hits10_filter/n_used_eval_triple
        #tail_meanrank_filter /= n_used_eval_triple
        #tail_hits10_filterf = tail_hits10_filter/n_used_eval_triple
        file.write("-----Filter-----\n")
        file.write("-----Head prediction-----\n")
        file.write("head_meanrank_filter \t head_hits10_filter \n")
        file.write(str(head_meanrank_filter))
        file.write("\t")
        file.write(str(head_hits10_filter))
        file.write("\n")
        file.write("-----Tail prediction-----\n")
        file.write("tail_meanrank_filter \t tail_hits10_filter \n")
        file.write(str(tail_meanrank_filter))
        file.write("\t")
        file.write(str(tail_hits10_filter))
        file.write("\n")
        file.close()
        print('-----Head prediction-----')
        print('MeanRank: {:.3f}'.format(head_meanrank_filter))
        print('Hits@10: {:.3f}'.format(head_hits10_filter))
        print('-----Tail prediction-----')
        print('MeanRank: {:.3f}'.format(tail_meanrank_filter))
        print('Hits@10: {:.3f}'.format(tail_hits10_filter))
        print('-----Average-----')
        print('MeanRank: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()


