import numpy as np

def motion_matching():
	from MyWindow import curFrame
	from bvh_handler import set_feature_vector, frame_list
	from utils import l2norm
	query_vector = "-0.08202883375533012 0.18560239097314008 -2.004747068122088 0.07568775119310739 0.18160711941731705 -1.981635260446206 -0.004426313251591277 -0.0035585629753850934 3.9245585179426e-05 -0.004443094511743101 -0.0032929316356127003 -0.0006082002896945493 0.0 0.0 0.0".split()

	#===motion matching===
	paths = []
	paths.append('features.txt')

	file = open(paths[0], 'r')

	nearest = 987654321
	nearest_index = -1
	for frame_feature in file:
		feat = frame_feature.split()
		index = int(feat[0])
		feat = feat[1:]
		temp = l2norm(np.array(query_vector, dtype = np.float32)- np.array(feat, dtype = np.float32))
		
		if nearest > temp:
			nearest = temp
			nearest_index = index

	# parsing bvh and set curFrame
	
	bvh_file = open("sample-walk.bvh", "r")
	curFrame1 = bvh_file.readlines()[nearest_index]
	print(nearest_index)
	print()
	print("curFrame")
	print(curFrame1)

motion_matching()
