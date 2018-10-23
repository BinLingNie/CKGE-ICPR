from __future__ import absolute_import

from dataset import KnowledgeGraph
from model import TransE

import tensorflow as tf
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default='../data/freebase_base/')
    parser.add_argument('--skp_file', type=str, default='../data/emd/freebase/ske_freebase_100_025_025-embs.npy')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--flag', type=bool, default=True)
    parser.add_argument('--save_file', type=str, default='../data/result/freebase/result_TransE_freebase_100-500-001-025-025-4096-embs.txt')

    args = parser.parse_args()
    print(args)

    kg = KnowledgeGraph(data_dir=args.data_dir)
    kge_model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                       score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate,
                       n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    saver = tf.train.Saver()

    with tf.Session(config=sess_config) as sess:

        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')

        #kge_model.check_norm(session=sess)
        summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph)
        if args.flag:
            for epoch in range(0, args.max_epoch):
                saver.save(sess, '../summary/TransE.ckpt')
                print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
                kge_model.launch_training(session=sess, summary_writer=summary_writer, skp_file=args.skp_file)
                if epoch%200 == 0 and epoch!=0:
                    kge_model.launch_evaluation(session=sess, test_data=kg.validation_triples, skp_file=args.skp_file,
                                                save_file=args.save_file)

            kge_model.launch_evaluation(session=sess, test_data= kg.test_triples, skp_file=args.skp_file, save_file=args.save_file)
            '''kge_model.launch_evaluation(session=sess, test_data=kg.one_to_one, skp_file=args.skp_file,
                                        save_file=args.save_file)
            kge_model.launch_evaluation(session=sess, test_data=kg.n_to_one, skp_file=args.skp_file,
                                        save_file=args.save_file)
            kge_model.launch_evaluation(session=sess, test_data=kg.one_to_n, skp_file=args.skp_file,
                                        save_file=args.save_file)
            kge_model.launch_evaluation(session=sess, test_data=kg.n_to_n, skp_file=args.skp_file,
                                        save_file=args.save_file)'''
        else:
            kge_model = saver.restore(sess, '../summary/TransE.ckpt')
            kge_model.launch_evaluation(session=sess)

if __name__ == '__main__':
    main()
