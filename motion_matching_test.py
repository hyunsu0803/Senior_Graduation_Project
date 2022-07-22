import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector


def QnA(key_input = None):

	tree_file = open('tree_dump.bin', 'rb')

	DB = pickle.load(tree_file)

	if key_input == "init":
		# temp_query = np.zeros((27,))
		temp_query = np.array([  0.          , 0.          , 0.          , 0.          , 0.,
   0.          , 0.          , 0.          , 0.         ,  0.,
   0.          , 0.         ,  0.2045277,   -1.84668984,   0.72330508,
   0.2043837,   -1.850461 ,    0.67703678 ,  6.13583114 , -55.40069508,
  21.69915251,   6.13151086, -55.51382989 , 20.31110331  , 0.,
   0.         ,  0.        ])		# first frame
	else:	
		temp_query = set_query_vector(key_input=key_input)
	
	# f = open("queryVectors.txt", 'a')
	# data = str(temp_query)+"\n"
	# f.write(data)
	# f.close()

	ans = DB.query(temp_query)
	qidx = ans[1]
	print("!!!!!!!! Query !!!!!!!!", temp_query)
	# print("qidx", qidx)
	print("!!!!!!!! DB feature !!!!!!!!", DB.data[qidx])

	bvh_name, nearest_frame_idx, FPS = utils.find_your_bvh(qidx)
	print("bvh name", bvh_name)
	print("nearest frame index", nearest_frame_idx)
	bvh_folder = './lafan1'
	bvh_path = os.path.join(bvh_folder, bvh_name)
	bvh_file = open(bvh_path, "r")

	coming_soon_10frames = bvh_file.readlines()
	print("lines length", len(coming_soon_10frames))
	coming_soon_10frames = coming_soon_10frames[nearest_frame_idx+1: nearest_frame_idx + 11]

	coming_soon_10frames = [i.split() for i in coming_soon_10frames]
	for i in range(len(coming_soon_10frames)):
		for j in range(len(coming_soon_10frames[i])):
			coming_soon_10frames[i][j] = float(coming_soon_10frames[i][j])

	return coming_soon_10frames, FPS


