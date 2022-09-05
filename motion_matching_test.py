import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector


def QnA(key_input = None):

	tree_file = open('tree_dump2.bin', 'rb')

	DB = pickle.load(tree_file)

	if key_input == "init":
		# query = np.ones((27,))
# 		temp_query = np.array([ 0.,          0.,          0.,          0.,          0.,          0.,
#   0.,          0.,          0.,          0.,          0.,          0.,
#   0.02596101, -0.44674884,  0.82710267, -1.75291079,  0.41621676, -0.24859644,
#  -0.30339763, -0.03691607, -0.27351549,  2.27067055, -0.26821648,  1.41159756,
#   2.15798027, -1.15139199,  0.88598802])
		query = np.array([ -0.06923568, -0.56889941, -0.16428767, -0.41164858, -0.09704712, -0.44940397,
  1.3418738,   0.0710877,   1.3495199,   0.06315013,  1.34948171,  0.11251585,
 -0.49796301, -0.82079598, -0.04171531,  0.08032729, -0.7272602,   0.78681163,
 -0.0451219,  -0.04894455, -0.237067,   -0.09413056,  0.03426322, -0.23647055,
 -0.02859922,  0.02325207, -0.19308389])
		query = np.zeros((27,))
	else:	
		query = set_query_vector(key_input=key_input)
		# print("real feature", temp_query)
# 		query = np.array([ -0.06923568, -0.56889941, -0.16428767, -0.41164858, -0.09704712, -0.44940397,
#   1.3418738,   0.0710877,   1.3495199,   0.06315013,  1.34948171,  0.11251585,
#  -0.49796301, -0.82079598, -0.04171531,  0.08032729, -0.7272602,   0.78681163,
#  -0.0451219,  -0.04894455, -0.237067,   -0.09413056,  0.03426322, -0.23647055,
#  -0.02859922,  0.02325207, -0.19308389])

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