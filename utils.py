import numpy as np


def l2norm(v):
    return np.sqrt(np.dot(v, v))


def normalized(v):
    l = l2norm(v)
    if l == 0:
        l = 1
    return 1 / l * np.array(v)


def exp(rv):
    th = l2norm(rv)
    costh = np.cos(th)
    sinth = np.sin(th)

    rv = normalized(rv)
    ux = rv[0]
    uy = rv[1]
    uz = rv[2]

    R = np.array(
        [[costh + ux * ux * (1 - costh), ux * uy * (1 - costh) - uz * sinth, ux * uz * (1 - costh) + uy * sinth],
         [uy * ux * (1 - costh) + uz * sinth, costh + uy * uy * (1 - costh), uy * uz * (1 - costh) - ux * sinth],
         [uz * ux * (1 - costh) - uy * sinth, uz * uy * (1 - costh) + ux * sinth, costh + uz * uz * (1 - costh)]])
    return R


def find_your_bvh(q):
    info_txt = open('db_index_info.txt', 'r')

    info = info_txt.readlines()
    info = [i.split() for i in info]
    for i in info:
        i[0] = int(i[0])
        i[2] = int(i[2])

    left_idx = 0
    right_idx = len(info)
    
    while(left_idx < right_idx):
        mid_idx = int((left_idx + right_idx) / 2)
        mid = info[mid_idx][0]

        if mid < q:
            left_idx = mid_idx + 1
        elif mid >= q:
            right_idx = mid_idx - 1

    ans = '@'
    mid = info[mid_idx][0]
    if mid <= q:
        ans = mid_idx
    else:
        ans = mid_idx - 1

    bvh_name = info[ans][1]
    bvh_line = q - info[ans][0] + info[ans][2]

    return bvh_name, bvh_line

        
# find_your_bvh(0)
# find_your_bvh(811)
# find_your_bvh(821)
# find_your_bvh(1030)
