import os
import pandas as pd
import numpy as np
import random

class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.entity_dict = {}
        self.relation_dict = {}
        self.n_entity = 0
        self.n_relation = 0

        self.training_triples = []  # list of triples in the form of (h, t, r)
        self.validation_triples = []
        self.test_triples = []
        self.one_to_one = []
        self.n_to_one = []
        self.one_to_n = []
        self.n_to_n = []

        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0

        self.load_dict()
        self.load_triples()

        self.golden_triple_pool = set(self.training_triples) | set(self.validation_triples) | set(self.test_triples)


    def load_dict(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.n_relation = len(self.relation_dict)
        print('#relation: {}'.format(self.n_relation))

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'newdev.txt'
        test_file = 'newtest.txt'
        #oneToone_file = '1-1-train.txt'
        #oneTon_file = '1-n-train.txt'
        #nToone_file = 'n-1-train.txt'
        #nTon_file = 'n-n-train.txt'

        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        self.training_triples = list(zip([self.entity_dict[h] for h in training_df[0]],
                                         [self.entity_dict[t] for t in training_df[2]],
                                         [self.relation_dict[r] for r in training_df[1]]))
        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))
        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        self.validation_triples = list(zip([self.entity_dict[h] for h in validation_df[0]],
                                           [self.entity_dict[t] for t in validation_df[2]],
                                           [self.relation_dict[r] for r in validation_df[1]]))
        self.n_validation_triple = len(self.validation_triples)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        self.test_triples = list(zip([self.entity_dict[h] for h in test_df[0]],
                                     [self.entity_dict[t] for t in test_df[2]],
                                     [self.relation_dict[r] for r in test_df[1]]))
        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))
        '''
        print('-----Loading one to one triples-----')
        oneToone_df = pd.read_table(os.path.join(self.data_dir, oneToone_file), header=None)
        self.one_to_one = list(zip([self.entity_dict[h] for h in oneToone_df[0]],
                                         [self.entity_dict[t] for t in oneToone_df[1]],
                                         [self.relation_dict[r] for r in oneToone_df[2]]))

        print('-----Loading one to n triples-----')
        oneTon_df = pd.read_table(os.path.join(self.data_dir, oneTon_file), header=None)
        self.one_to_n = list(zip([self.entity_dict[h] for h in oneTon_df[0]],
                                   [self.entity_dict[t] for t in oneTon_df[1]],
                                   [self.relation_dict[r] for r in oneTon_df[2]]))

        print('-----Loading n to one triples-----')
        nToone_df = pd.read_table(os.path.join(self.data_dir, nToone_file), header=None)
        self.n_to_one = list(zip([self.entity_dict[h] for h in nToone_df[0]],
                                 [self.entity_dict[t] for t in nToone_df[1]],
                                 [self.relation_dict[r] for r in nToone_df[2]]))

        print('-----Loading n to n triples-----')
        nTon_df = pd.read_table(os.path.join(self.data_dir, nTon_file), header=None)
        self.n_to_n = list(zip([self.entity_dict[h] for h in nTon_df[0]],
                                 [self.entity_dict[t] for t in nTon_df[1]],
                                 [self.relation_dict[r] for r in nTon_df[2]]))'''

    def next_raw_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_training_triple)
        start = 0
        while start < self.n_training_triple:
            end = min(start + batch_size, self.n_training_triple)
            yield [self.training_triples[i] for i in rand_idx[start:end]], [i for i in rand_idx[start:end]]
            start = end

    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_all_pos = raw_batch
                batch_pos = batch_all_pos[0]
                batch_triple_id = batch_all_pos[1]
                batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)

                for head, tail, relation in batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.sample(list(self.entity_dict.values()), 1)[0]
                        else:
                            tail_neg = random.sample(list(self.entity_dict.values()), 1)[0]
                        if (head_neg, tail_neg, relation) not in self.golden_triple_pool:
                            break
                    batch_neg.append((head_neg, tail_neg, relation))

                batch_tripos = []
                batch_trineg = []

                for i in range(0, len(batch_triple_id)):
                    batch_tripos.append(batch_pos[i])
                    batch_trineg.append(batch_neg[i])

                out_queue.put((batch_tripos, batch_trineg))
