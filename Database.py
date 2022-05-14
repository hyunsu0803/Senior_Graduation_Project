import numpy as np
from scipy.spatial.distance import sqeuclidean


class Database:
    def __init__(self):
        self.data = []    # column : feature 27 + "bvh file name" + bvh_line
        pass

    def buildDB(self, feat_txt):   
        # txt file �씫怨� split�븯怨� float 蹂��솚源뚯�� �떎 留덉튂怨� self.data�뿉 feature�뱾 �쟾遺� �삱由� 
        feats = feat_txt.readlines()
        for feat in feats:
            feat = feat.split()
            feat = [float(i) for i in feat[:27]]
            feat.append(feat[27])
            feat.append(int(feat[28]))

            self.data.append(feat)

        print("How many frames :", len(self.data))

    def query(self, query_vec):
        # self.data瑜� �쟾遺� �룎硫댁꽌 query�쓽 nearest 李얠쓬.
        # 李얠�� frame�쓽 bvh_name, bvh_line�쓣 return. 

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

        return bvh_name, bvh_line   # return 諛쏆�� 怨녹뿉�꽌 �븣�븘�꽌 bvh file濡� 媛�寃좎��? �뼱�뵒�꽌 return 諛쏅깘

