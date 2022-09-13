import numpy as np
import pickle
import utils
import os
from bvh_handler import set_query_vector

global past_real_query
past_real_query = np.zeros((27,))


def QnA(key_input = None):
	global past_real_query

	tree_file = open('tree_dump.bin', 'rb')

	DB = pickle.load(tree_file)
	made_query = np.zeros((27,))

	if key_input == "init":
		# query = np.array([ 2.68290149e+00,  3.17010629e+01,  1.29321473e+00,  6.06929369e+01,
		# 	-3.72086041e+00, 9.42400043e+01, -1.61090305e-02,  9.99870241e-01,
		# 	6.54336934e-02,  9.97856919e-01,  7.94921304e-02,  9.96835494e-01,
		# 	9.08802483e+00,  9.96929501e+00,  3.95038431e+01, -3.60876742e+00,
		# 	8.35734600e+00, -2.52759112e+01, -2.71575568e+01, -8.45155278e+00,
		# 	1.15894315e+02, -1.97930926e+00,  1.15349951e+01,  8.08864516e+00,
		# 	1.71434697e+00, -1.52801514e+01,  9.79888749e+01])
		query = np.array([ 2.96960973e-01,  9.96564847e+00, -4.50558696e+00,  3.36485732e+01,
		-7.10655357e+00,  6.16891943e+01,  5.23026731e-02,  9.98631278e-01,
		5.05232841e-02,  9.98722883e-01, -3.49997476e-03,  9.99993875e-01,
		5.86174130e+00,  6.39006306e+00, -7.39693220e+00, -1.58474458e+01,
		1.36000753e+01,  3.77870203e+00,  8.16262709e-02,  9.95738551e-01,
		-5.71020553e-02,  2.73323919e+01, -1.12751758e+01,  9.64274966e+01,
		5.65231538e+00,  3.75823975e-01,  1.62885360e+01])

		# query = np.zeros((27,))
	else:	
		made_query = set_query_vector(key_input=key_input)
		query = past_real_query
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

	print("!!!!!!!! Query !!!!!!!!", made_query, sep='\n')
	print("!!!!!!!! DB feature !!!!!!!!", DB.data[qidx], sep='\n')
	
	print("!!!!!!! + 10 index feature !!!!!!!", past_real_query, sep = '\n')
	past_real_query = DB.data[qidx+10]

	print("############query feature difference##############")
	print(np.linalg.norm(made_query - np.array(DB.data[qidx])))
	print("############query feature difference vector##############")
	print(made_query - np.array(DB.data[qidx]))

	# print("############query feature difference vector##############")
	# print(query - made_query)

	bvh_folder = './lafan1'
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
    info_txt = open('db_index_info.txt', 'r')

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