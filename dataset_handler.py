import numpy as np
from Joint import Joint
from Feature import Feature

from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation as R
import os
from scipy.spatial import cKDTree
import pickle


class status:
    joint_list = []
    frame_list = []
    feature_list = []
    timeStep = 0.2
    line_index = 0
    curFrame = []
    futureFrames = [None, None, None]
    futurePosition = [[None], [None], [None]]
    futureDirection = [[None], [None], [None]]


def parsing_bvh(bvh):
    frameTime = 0
    status.frame_list = []
    status.line_index = 0

    num_of_frames = 0

    line = bvh.readline().split()
    status.line_index += 1

    if line[0] == 'HIERARCHY':
        line = bvh.readline().split()
        status.line_index += 1
        if line[0] == 'ROOT':
            status.joint_list = []
            buildJoint(bvh, line[1])  # build ROOT and other joints

    line = bvh.readline().split()
    status.line_index += 1

    if line[0] == 'MOTION':
        line = bvh.readline().split()
        status.line_index += 1
        if line[0] == 'Frames:':
            num_of_frames = int(line[1])
        line = bvh.readline().split()
        status.line_index += 1

        if line[0] == 'Frame' and line[1] == 'Time:':
            frameTime = float(line[2])

        for i in range(num_of_frames):
            line = bvh.readline().split()
            line = list(map(float, line))
            status.frame_list.append(line)
        last = [0] * len(status.frame_list[0])   
        status.frame_list.append(last)

    FPS = int(1 / frameTime)

    return num_of_frames, FPS


def buildJoint(bvh, joint_name):

    line = bvh.readline().split()  # remove '{'
    status.line_index += 1
    newJoint = Joint(joint_name)

    # check if it's foot joint
    if "Foot" in joint_name:
        newJoint.set_is_foot(True)

    newJoint.set_index(len(status.joint_list))

    status.joint_list.append(newJoint)

    line = bvh.readline().split()
    status.line_index += 1
    if line[0] == 'OFFSET':
        offset = np.array(list(map(float, line[1:])), dtype='float32')
        if np.sqrt(np.dot(offset, offset)) > Joint.resize:
            Joint.resize = np.sqrt(np.dot(offset, offset))
        newJoint.set_offset(offset)

    line = bvh.readline().split()
    status.line_index += 1
    if line[0] == 'CHANNELS':
        newJoint.set_channel(line[2:])

    while True:
        line = bvh.readline().split()
        status.line_index += 1
        if line[0] == 'JOINT':
            newJoint.append_child_joint(buildJoint(bvh, line[1]))

        elif line[0] == 'End' and line[1] == 'Site':
            line = bvh.readline().split()  # remove '{'
            status.line_index += 1
            line = bvh.readline().split()
            status.line_index += 1
            if line[0] == 'OFFSET':
                offset = np.array(list(map(float, line[1:])), dtype='float32')
                if np.sqrt(np.dot(offset, offset)) > Joint.resize:
                    Joint.resize = np.sqrt(np.dot(offset, offset))
                newJoint.set_end_site(offset)
            line = bvh.readline().split()  # remove '}'
            status.line_index += 1

        elif line[0] == '}':
            return newJoint


def set_joint_feature(joint, parentMatrix, rootMatrix=None):
    newMatrix = np.identity(4)
    # cur_position = [0, 0, 0, 1]

    # get current joint's offset from parent joint
    curoffset = joint.get_offset() / Joint.resize

    temp = np.identity(4)
    temp[:3, 3] = curoffset
    newMatrix = newMatrix @ temp

    # channel rotation
    # ROOT
    if len(joint.get_channel()) == 6:
        ROOTPOSITION = np.array(status.curFrame[:3], dtype='float32')
        ROOTPOSITION /= Joint.resize

        # move root's transformation matrix's origin using translation data
        temp = np.identity(4)
        temp[:3, 3] = ROOTPOSITION
        newMatrix = newMatrix @ temp

        for i in range(3, 6):
            if joint.get_channel()[i].upper() == 'XROTATION':
                xr = status.curFrame[i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx
            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = status.curFrame[i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry
            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = status.curFrame[i]
                zr = np.radians(zr)
                Rz = np.array([[np.cos(zr), -np.sin(zr), 0, 0],
                               [np.sin(zr), np.cos(zr), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
                newMatrix = newMatrix @ Rz

    # JOINT
    else:
        index = joint.get_index()
        for i in range(3):
            if joint.get_channel()[i].upper() == 'XROTATION':
                xr = status.curFrame[(index + 1) * 3 + i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx

            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = status.curFrame[(index + 1) * 3 + i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry

            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = status.curFrame[(index + 1) * 3 + i]
                zr = np.radians(zr)
                Rz = np.array([[np.cos(zr), -np.sin(zr), 0, 0],
                               [np.sin(zr), np.cos(zr), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
                newMatrix = newMatrix @ Rz

    joint.set_transform_matrix(parentMatrix @ newMatrix)
    transform_matrix = joint.get_transform_matrix()
    global_position = transform_matrix @ np.array([0., 0., 0., 1.])

    # set parent's global position (if it is root joint, parent_position is current_position)
    if joint.get_is_root():
        parent_position = global_position
        status.futurePosition = [None, None, None]
        status.futureDirection = [None, None, None]

        for i in range(0, 3):
            temp = np.array([0, 0, 0, 1])
            temp[:3] = status.futureFrames[i][:3]
            status.futurePosition[i] = np.linalg.inv(transform_matrix) @ temp

            status.futurePosition[i] = status.futurePosition[i][0::2]
            
            default_facing_direction = np.array([1., 0., 0.])
            rotation_current = R.from_euler('zxy', status.curFrame[3:6], degrees=True)
            rotation_future = R.from_euler('zxy', status.futureFrames[i][3:6], degrees=True)
            rotation_current = np.array(rotation_current.as_matrix())
            rotation_future = np.array(rotation_future.as_matrix())
            status.futureDirection[i] = R.from_matrix(rotation_current.T @ rotation_future).as_matrix() @ default_facing_direction
            status.futureDirection[i] = status.futureDirection[i][0::2]

    else:
        parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

    cur_position = global_position

    # Check if it's Root joint, otherwise update Joint class's data
    # velocity, rotation velocity update

    if joint.get_is_root():
        rootMatrix = joint.get_transform_matrix()

    else:
        # get root local position and root local velocity
        new_root_local_position = (rootMatrix.T @ global_position)[:3]  # local to root joint
        past_root_local_position = joint.get_root_local_position()  # local to root joint
        root_local_velocity = ((new_root_local_position - past_root_local_position) / status.timeStep)

        # get root local rotation and root local angular velocity
        new_root_local_rotation_matrix = (rootMatrix.T @ transform_matrix)[:3, :3]
        r = R.from_matrix(new_root_local_rotation_matrix)
        new_root_local_rotation = np.array(r.as_quat())
        past_root_local_rotation = joint.get_root_local_rotation()

        # set joint class's value
        joint.set_global_position(global_position[:3])
        joint.set_root_local_velocity(root_local_velocity)
        joint.set_root_local_position(new_root_local_position[:3])
        joint.set_root_local_rotation(new_root_local_rotation)


    if joint.end_site is None:
        for j in joint.get_child():
            set_joint_feature(j, joint.get_transform_matrix(), rootMatrix)


def set_feature_vector(feature_vector):
    two_foot_position = []
    two_foot_velocity = []
    hip_velocity = []

    for joint in status.joint_list:
        if joint.get_is_root():
            hip_velocity.append(joint.get_root_local_velocity())
        elif joint.get_is_foot():
            two_foot_position.append(joint.get_root_local_position())
            two_foot_velocity.append(joint.get_root_local_velocity())

    feature_vector.set_future_position(np.array(status.futurePosition).reshape(6, ))
    feature_vector.set_future_direction(np.array(status.futureDirection).reshape(6, ))

    # test future info setting
    # feature_vector.set_future_position(np.zeros_like(futurePosition).reshape(6, ))
    # feature_vector.set_future_direction(np.zeros_like(futureDirection).reshape(6, ))
    feature_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
    feature_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
    feature_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))


def main():
    status.curFrame = []
    Joint.resize = 1

    bvh_dir = './lafan2/'
    bvh_names = os.listdir(bvh_dir)
    bvh_names.sort()

    with open('db_index_info2.txt', 'w') as db_index_info:

        db_index = 0
        data = []
        for bvh_name in bvh_names:
            bvh_path = os.path.join(bvh_dir, bvh_name)
            bvh = open(bvh_path, 'r')

            # db_index_info.txt 
            num_of_frames, FPS = parsing_bvh(bvh)
            info = ' '.join([str(db_index), bvh_name, str(status.line_index), str(FPS)+'\n'])
            db_index_info.write(info)

            db_index += num_of_frames - FPS 

            for i in range(0, len(status.frame_list) - FPS -1):
                status.curFrame = status.frame_list[i]
                status.futureFrames = []  
                for j in [int(FPS/3), int(FPS/3*2), int(FPS)]:    
                    if i + j < len(status.frame_list):
                        status.futureFrames.append(status.frame_list[i + j])
                set_joint_feature(status.joint_list[0], np.identity(4), None)
                feature_vector = Feature()
                set_feature_vector(feature_vector)
                status.feature_list.append(feature_vector)

                data.append(feature_vector.get_feature_list())

    DB = cKDTree(np.array(data))
    with open('tree_dump2.bin', 'wb') as dump_file:
        pickle.dump(DB, dump_file)

    # with open('tree_dump2.bin', 'rb') as dump_file:
    #     ddbb = pickle.load(dump_file)
        # temp_query = np.zeros((27,))
        # result = ddbb.query(temp_query)
        # print(result)
        # print(ddbb.data.shape)  


if __name__ == "__main__":
    main()
