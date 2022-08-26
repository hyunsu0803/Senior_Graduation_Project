import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector


def QnA(key_input = None):

	tree_file = open('tree_dump2.bin', 'rb')

	DB = pickle.load(tree_file)

	if key_input == "init":
		# temp_query = np.((27,))
		temp_query = np.array([-0.12462778, -0.06519135, -0.03598204, -0.3029604,   0.02616999, -0.38266711 ,
  0.57028826,  1.49323846,  0.58805032,  1.47388588,  0.57100566,  1.48433671,
 -0.54135482, -0.72427646, -0.08090772, -0.00561489, -0.34744219, -0.05211065,
 -0.04538396, -0.0198117,  -0.23418385, -0.09717699,  0.02300373,  0.10834213,
 -0.07682479, -0.06230886, -0.07244505])
	else:	
		temp_query = set_query_vector(key_input=key_input)

	ans = DB.query(temp_query)
	qidx = ans[1]
	print("!!!!!!!! Query !!!!!!!!", temp_query)
	print("!!!!!!!! DB feature !!!!!!!!", DB.data[qidx])

	print("############query feature difference##############")
	print(np.linalg.norm(temp_query - np.array(DB.data[qidx])))
	print("############query feature difference vector##############")
	print(temp_query - np.array(DB.data[qidx]))

	bvh_name, nearest_frame_idx, FPS = utils.find_your_bvh(qidx)
	print("bvh name", bvh_name, nearest_frame_idx)

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


