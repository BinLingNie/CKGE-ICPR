import os
import numpy as np
triple_path = np.load("../data/FB15k/newtriple_path.npy")
triple_flag = np.load("../data/FB15k/path_flag.npy")
def get_path(triple_id, triple_pos, triple_neg):
	print('start')
	path_list = []
	path_triple_pos = []
	path_triple_neg = []
	single_triple_pos = []
	single_triple_neg = []

	for i in range(0, len(triple_id)):
		if triple_flag[triple_id[i]]:
			single_triple_pos.append(triple_pos[i])
			single_triple_neg.append(triple_neg[i])
		else:
			path_list.append(triple_path[triple_id[i]])
			path_triple_pos.append(triple_pos[i])
			path_triple_neg.append(triple_neg[i])
	return path_list, path_triple_pos, path_triple_neg, single_triple_pos, single_triple_neg