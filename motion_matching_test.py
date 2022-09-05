import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector


def QnA(key_input = None):

	tree_file = open('tree_dump2.bin', 'rb')

	DB = pickle.load(tree_file)

	if key_input == "init":
		query = np.ones((27,))
# 		temp_query = np.array([ 0.,          0.,          0.,          0.,          0.,          0.,
#   0.,          0.,          0.,          0.,          0.,          0.,
#   0.02596101, -0.44674884,  0.82710267, -1.75291079,  0.41621676, -0.24859644,
#  -0.30339763, -0.03691607, -0.27351549,  2.27067055, -0.26821648,  1.41159756,
#   2.15798027, -1.15139199,  0.88598802])
		query = np.array([ 0.,          0.,          0.,          0.,          0.,          0.,
  					  0.,          0.,          0.,          0.,          0.,          0.,
 					-0.61352249, -0.10848645, -0.32916056, -0.25029716, -0.15555006,  0.54588994,
 					-0.0300607,  0.21631076, -0.21581551, -0.1518274,  -1.6671526,   1.12778864,
 					-0.15093671,  0.04813088,  0.12306743])
	else:	
		query = set_query_vector(key_input=key_input)
		# print("real feature", temp_query)
		# temp_query = np.array([ 0.,          0.,          0.,          0.,          0.,          0.,
  		# 			  0.,          0.,          0.,          0.,          0.,          0.,
 		# 			-0.61352249, -0.10848645, -0.32916056, -0.25029716, -0.15555006,  0.54588994,
 		# 			-0.0300607,  0.21631076, -0.21581551, -0.1518274,  -1.6671526,   1.12778864,
 		# 			-0.15093671,  0.04813088,  0.12306743])

	ans = DB.query(query)
	qidx = ans[1]

	bvh_name, nearest_frame_idx, FPS = find_your_bvh(qidx)
	print()
	print("bvh name", bvh_name, nearest_frame_idx)

	print("!!!!!!!! Query !!!!!!!!", query, sep='\n')
	print("!!!!!!!! DB feature !!!!!!!!", DB.data[qidx], sep='\n')

	print("############query feature difference##############")
	print(np.linalg.norm(query - np.array(DB.data[qidx])))
	print("############query feature difference vector##############")
	print(query - np.array(DB.data[qidx]))

	bvh_folder = './lafan2'
	bvh_path = os.path.join(bvh_folder, bvh_name)
	bvh_file = open(bvh_path, "r")

	coming_soon_10frames = bvh_file.readlines()
	future_10frame = coming_soon_10frames[nearest_frame_idx + 20]
	future_20frame = coming_soon_10frames[nearest_frame_idx + 30]
	future_30frame = coming_soon_10frames[nearest_frame_idx + 40]

	coming_soon_10frames = coming_soon_10frames[nearest_frame_idx+1: nearest_frame_idx + 11]

	

	coming_soon_10frames = [i.split() for i in coming_soon_10frames]
	future_10frame = future_10frame.split()
	future_20frame = future_20frame.split()
	future_30frame = future_30frame.split()

	for i in range(len(coming_soon_10frames)):
		for j in range(len(coming_soon_10frames[i])):
			coming_soon_10frames[i][j] = float(coming_soon_10frames[i][j])

	for i in range(len(future_10frame)):
		future_10frame[i] = float(future_10frame[i])
		future_20frame[i] = float(future_20frame[i])
		future_30frame[i] = float(future_30frame[i])


	return coming_soon_10frames, FPS , [future_10frame, future_20frame, future_30frame]


def find_your_bvh(q):
    info_txt = open('db_index_info2.txt', 'r')

    info = info_txt.readlines()
    info = [i.split() for i in info]
    for i in info:
        i[0] = int(i[0])
        i[2] = int(i[2])

    best = info[-1]
    for i in range(len(info) - 1):
        if info[i][0] <= q and info[i+1][0] > q:
            best = info[i]
    
    bvh_name = best[1]
    bvh_line = q - best[0] + best[2]
    FPS = best[-1]

    return bvh_name, bvh_line, int(FPS)