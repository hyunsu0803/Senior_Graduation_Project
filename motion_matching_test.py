import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector


def QnA(key_input = None):

	tree_file = open('tree_dump2.bin', 'rb')

	DB = pickle.load(tree_file)

	if key_input == "init":
		# temp_query = np.zeros((27,))
		temp_query = np.array([ 0.          ,0.          ,0.          ,0.          ,0.          ,0.,
  0.          ,0.          ,0.          ,0.          ,0.          ,0.,
  0.18384297,  1.69866049,  0.10729861,  0.18609492,  1.79979874,  0.05209561,
 -0.02820048,  0.08129325, -0.72126381, -0.02636612,  0.12200875, -0.76057781,
  0.        ,  0.        ,  0.        ])
	else:	
		temp_query = set_query_vector(key_input=key_input)
	
	f = open("queryVectors.txt", 'a')
	data = str(temp_query)+"\n"
	f.write(data)
	f.close()

	ans = DB.query(temp_query)
	qidx = ans[1]
	print("qidx", qidx)
	print("!!!!!!!! DB feature !!!!!!!!", DB.data[qidx])

	bvh_name, nearest_frame_idx, FPS = utils.find_your_bvh(qidx)
	print(bvh_name)
	bvh_folder = './lafan2'
	bvh_path = os.path.join(bvh_folder, bvh_name)
	bvh_file = open(bvh_path, "r")

	coming_soon_10frames = bvh_file.readlines()
	coming_soon_10frames = coming_soon_10frames[nearest_frame_idx+1: nearest_frame_idx + 11]

	coming_soon_10frames = [i.split() for i in coming_soon_10frames]
	for i in range(len(coming_soon_10frames)):
		for j in range(len(coming_soon_10frames[i])):
			coming_soon_10frames[i][j] = float(coming_soon_10frames[i][j])

	return coming_soon_10frames, FPS


