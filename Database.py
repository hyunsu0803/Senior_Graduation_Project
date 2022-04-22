import numpy as np
from scipy.spatial.distance import sqeuclidean


class Database:
    def __init__(self):
        self.data = []    # column : feature 27 + "bvh file name" + bvh_line
        pass

    def buildDB(self, feat_txt):   
        # txt file 읽고 split하고 float 변환까지 다 마치고 self.data에 feature들 전부 올림 
        feats = feat_txt.readlines()
        for feat in feats:
            feat = feat.split()
            feat = [float(i) for i in feat[:27]]
            feat.append(feat[27])
            feat.append(int(feat[28]))

            self.data.append(feat)

        print("How many frames :", len(self.data))

    def query(self, query_vec):
        # self.data를 전부 돌면서 query의 nearest 찾음.
        # 찾은 frame의 bvh_name, bvh_line을 return. 

        min_dist = 987654321

        bvh_name = None
        bvh_line = None

        for frame in self.data:
            feature = np.array(frame[:27])
            dist = sqeuclidean(query_vec, feature)
            if dist < min_dist:
                min_dist = dist
                bvh_name = frame[27]
                bvh_line = frame[28]

        return bvh_name, bvh_line   # return 받은 곳에서 알아서 bvh file로 가겠지? 어디서 return 받냐

