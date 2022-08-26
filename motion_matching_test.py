import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector


def QnA(key_input = None):

	tree_file = open('tree_dump.bin', 'rb')

	DB = pickle.load(tree_file)

	if key_input == "init":
		temp_query = np.zeros((27,))
	else:	
		temp_query = set_query_vector(key_input=key_input)

	ans = DB.query(temp_query)
	qidx = ans[1]
	print("!!!!!!!! Query !!!!!!!!", temp_query)
	print("!!!!!!!! DB feature !!!!!!!!", DB.data[qidx])

	print()
	print("############query feature difference##############")
	print(np.linalg.norm(temp_query - np.array(DB.data[qidx])))
	print("############query feature difference vector##############")
	print(temp_query - np.array(DB.data[qidx]))
	# print()

	bvh_name, nearest_frame_idx, FPS = utils.find_your_bvh(qidx)
	print("bvh name", bvh_name, nearest_frame_idx)

	bvh_folder = './lafan1'
	bvh_path = os.path.join(bvh_folder, bvh_name)
	bvh_file = open(bvh_path, "r")

	coming_soon_10frames = bvh_file.readlines()
	future_10frame = coming_soon_10frames[nearest_frame_idx + 20]
	future_20frame = coming_soon_10frames[nearest_frame_idx + 30]
	future_30frame = coming_soon_10frames[nearest_frame_idx + 40]

	coming_soon_10frames = coming_soon_10frames[nearest_frame_idx+1: nearest_frame_idx + 11]
	
	# coming_soon_10frames = coming_soon_10frames[nearest_frame_idx+0: nearest_frame_idx + 10]

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


