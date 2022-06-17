import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector


def QnA(key_input = None):

	tree_file = open('tree_dump2.bin', 'rb')

	DB = pickle.load(tree_file)

	if key_input == "init":
		temp_query = np.zeros((27,))
	else:	
		temp_query = set_query_vector(key_input=key_input)

	print("temp query", temp_query)

	ans = DB.query(temp_query)
	qidx = ans[1]
	print("qidx", qidx)
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


