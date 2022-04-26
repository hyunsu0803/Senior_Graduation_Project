import numpy as np
import pickle
import utils
import os


def motion_matching():
    # from MyWindow import curFrame
    #from bvh_handler import set_feature_vector, frame_list
    from utils import l2norm
    query_vector = "-0.08202883375533012 0.18560239097314008 -2.004747068122088 0.07568775119310739 0.18160711941731705 -1.981635260446206 -0.004426313251591277 -0.0035585629753850934 3.9245585179426e-05 -0.004443094511743101 -0.0032929316356127003 -0.0006082002896945493 0.0 0.0 0.0".split()

    # ===motion matching===
    paths = ['features.txt']

    txt_file = open(paths[0], 'r')

    nearest = 987654321
    nearest_index = -1

    frame_features = txt_file.readlines()
    for i in range(len(frame_features)):
        frame_feature = frame_features[i]
        feat = frame_feature.split()
        index = int(feat[0])
        feat = feat[1:]
        temp = l2norm(np.array(query_vector, dtype=np.float32) - np.array(feat, dtype=np.float32))

        if nearest > temp:
            nearest = temp
            nearest_index = index

    # parsing bvh and set curFrame

    bvh_file = open("sample-walk.bvh", "r")

    coming_soon_50frames = bvh_file.readlines()
    if nearest_index + 50 < len(coming_soon_50frames):
        coming_soon_50frames = coming_soon_50frames[nearest_index: nearest_index + 50]
    else:
        coming_soon_50frames = coming_soon_50frames[nearest_index:-1]

    coming_soon_50frames = [i.split() for i in coming_soon_50frames]
    for i in range(len(coming_soon_50frames)):
        for j in range(len(coming_soon_50frames[i])):
            coming_soon_50frames[i][j] = float(coming_soon_50frames[i][j])

    # print(nearest_index)
    # print()
    # print("curFrame")
    # print(coming_soon_10frames)

    return coming_soon_50frames



def QnA():
    tree_file = open('tree_dump.bin', 'rb')
    DB = pickle.load(tree_file)

    temp_query = np.zeros((27,))
    ans = DB.query(temp_query)
    print(ans)
    qidx = ans[1]

    bvh_name, nearest_frame_idx = utils.find_your_bvh(qidx)
    bvh_folder = 'bvh_folder'
    bvh_path = os.path.join(bvh_folder, bvh_name)
    bvh_file = open(bvh_path, "r")

    print(bvh_path)

    coming_soon_50frames = bvh_file.readlines()
    if nearest_frame_idx + 50 < len(coming_soon_50frames):
        coming_soon_50frames = coming_soon_50frames[nearest_frame_idx: nearest_frame_idx + 50]
    else:
        coming_soon_50frames = coming_soon_50frames[nearest_frame_idx:]

    coming_soon_50frames = [i.split() for i in coming_soon_50frames]
    for i in range(len(coming_soon_50frames)):
        for j in range(len(coming_soon_50frames[i])):
            coming_soon_50frames[i][j] = float(coming_soon_50frames[i][j])

    print(coming_soon_50frames)
    exit()

    return coming_soon_50frames


QnA()