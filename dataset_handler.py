import numpy as np
from Joint import Joint
from Feature import Feature

from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation as R
import os
from scipy.spatial import cKDTree
import pickle


class state:
    joint_list = []
    frame_list = []
    feature_list = []
    line_index = 0
    curFrame = []
    futureFrames = [None, None, None]
    local_futurePosition = [[None], [None], [None]]
    futureDirection = [[None], [None], [None]]


def parsing_bvh(bvh):
    frameTime = 0
    state.frame_list = []
    state.line_index = 0

    num_of_frames = 0

    line = bvh.readline().split()
    state.line_index += 1

    if line[0] == 'HIERARCHY':
        line = bvh.readline().split()
        state.line_index += 1
        if line[0] == 'ROOT':
            state.joint_list = []
            buildJoint(bvh, line[1])  # build ROOT and other joints
            Joint.resize = int(Joint.resize)

    line = bvh.readline().split()
    state.line_index += 1

    if line[0] == 'MOTION':
        line = bvh.readline().split()
        state.line_index += 1
        if line[0] == 'Frames:':
            num_of_frames = int(line[1])
        line = bvh.readline().split()
        state.line_index += 1

        if line[0] == 'Frame' and line[1] == 'Time:':
            frameTime = float(line[2])

        for i in range(num_of_frames):
            line = bvh.readline().split()
            line = list(map(float, line))
            state.frame_list.append(line)
        last = [0] * len(state.frame_list[0])   
        state.frame_list.append(last)

    FPS = int(1 / frameTime)

    return num_of_frames, FPS


def buildJoint(bvh, joint_name):

    line = bvh.readline().split()  # remove '{'
    state.line_index += 1
    newJoint = Joint(joint_name)

    # check if it's foot joint
    if "Foot" in joint_name:
        newJoint.set_is_foot(True)

    newJoint.set_index(len(state.joint_list))

    state.joint_list.append(newJoint)

    line = bvh.readline().split()
    state.line_index += 1
    if line[0] == 'OFFSET':
        offset = np.array(list(map(float, line[1:])), dtype='float32')
        if joint_name != "Hips" and np.sqrt(np.dot(offset, offset)) > Joint.resize:
            Joint.resize = np.sqrt(np.dot(offset, offset))
        newJoint.set_offset(offset)

    line = bvh.readline().split()
    state.line_index += 1
    if line[0] == 'CHANNELS':
        newJoint.set_channel(line[2:])

    while True:
        line = bvh.readline().split()
        state.line_index += 1
        if line[0] == 'JOINT':
            newJoint.append_child_joint(buildJoint(bvh, line[1]))

        elif line[0] == 'End' and line[1] == 'Site':
            line = bvh.readline().split()  # remove '{'
            state.line_index += 1
            line = bvh.readline().split()
            state.line_index += 1
            if line[0] == 'OFFSET':
                offset = np.array(list(map(float, line[1:])), dtype='float32')
                if joint_name != "Hips" and np.sqrt(np.dot(offset, offset)) > Joint.resize:
                    Joint.resize = np.sqrt(np.dot(offset, offset))
                newJoint.set_end_site(offset)
            line = bvh.readline().split()  # remove '}'
            state.line_index += 1

        elif line[0] == '}':
            return newJoint


def set_joint_feature(joint, parentMatrix, rootMatrix=None):
    newMatrix = np.identity(4)

    # get current joint's offset from parent joint
    curoffset = joint.get_offset() / Joint.resize

    temp = np.identity(4)
    if len(joint.get_channel()) != 6:
        temp[:3, 3] = curoffset
    newMatrix = newMatrix @ temp

    # channel rotation
    # ROOT
    if len(joint.get_channel()) == 6:
        ROOTPOSITION = np.array(state.curFrame[:3], dtype='float32')
        ROOTPOSITION /= Joint.resize

        # move root's transformation matrix's origin using translation data
        temp = np.identity(4)
        temp[:3, 3] = ROOTPOSITION
        newMatrix = newMatrix @ temp

        for i in range(3, 6):
            if joint.get_channel()[i].upper() == 'XROTATION':
                xr = state.curFrame[i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx
            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = state.curFrame[i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry
            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = state.curFrame[i]
                zr = np.radians(zr)
                Rz = np.array([[np.cos(zr), -np.sin(zr), 0., 0.],
                               [np.sin(zr), np.cos(zr), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype="float32")
                
                newMatrix = newMatrix @ Rz

    # JOINT
    else:
        index = joint.get_index()
        for i in range(3):
            if joint.get_channel()[i].upper() == 'XROTATION':
                xr = state.curFrame[(index + 1) * 3 + i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx

            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = state.curFrame[(index + 1) * 3 + i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry

            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = state.curFrame[(index + 1) * 3 + i]
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
        # parent_position = global_position
        state.local_futurePosition = [None, None, None]
        state.futureDirection = [None, None, None]

        for i in range(0, 3):

            global_root_position = np.array([0., 0., 0., 1.])
            global_root_position[:3] = state.futureFrames[i][:3]
            global_root_position[:3] /= Joint.resize

            state.local_futurePosition[i] = np.linalg.inv(transform_matrix) @ global_root_position            
            state.local_futurePosition[i] = state.local_futurePosition[i][0::2]

            
            default_facing_direction = np.array([1., 0., 0.])
            global_rotation_current = R.from_euler('zyx', state.curFrame[3:6], degrees=True).as_matrix()
            global_rotation_future = R.from_euler('zyx', state.futureFrames[i][3:6], degrees=True).as_matrix()

            global_future_direction = global_rotation_future @ default_facing_direction
            local_future_direction = global_rotation_current.T @ global_future_direction
            state.futureDirection[i] = local_future_direction[0::2]

        # exit()

    # Check if it's Root joint, otherwise update Joint class's data
    # velocity, rotation velocity update

    if joint.get_is_root():
        rootMatrix = joint.get_transform_matrix()

    else:
        # get root local position and root local velocity
        new_root_local_position = (np.linalg.inv(rootMatrix) @ global_position)[:3]  # local to root joint
        past_root_local_position = joint.get_root_local_position()  # local to root joint
        root_local_velocity = ((new_root_local_position - past_root_local_position) * 30)
   

        # get root local rotation and root local angular velocity
        new_root_local_rotation_matrix = (np.linalg.inv(rootMatrix) @ transform_matrix)[:3, :3]
        r = R.from_matrix(new_root_local_rotation_matrix)
        new_root_local_rotation = np.array(r.as_quat())

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

    for joint in state.joint_list:
        if joint.get_is_root():
            hip_velocity.append(joint.get_root_local_velocity())
        elif joint.get_is_foot():
            two_foot_position.append(joint.get_root_local_position())
            two_foot_velocity.append(joint.get_root_local_velocity())

    feature_vector.set_future_position(np.array(state.local_futurePosition).reshape(6, ))
    feature_vector.set_future_direction(np.array(state.futureDirection).reshape(6, ))

    # test future info setting
    # feature_vector.set_future_position(np.zeros_like(state.futurePosition).reshape(6, ))
    # feature_vector.set_future_direction(np.zeros_like(state.futureDirection).reshape(6, ))
    feature_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
    feature_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
    feature_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))


def main():
    state.curFrame = []

    bvh_dir = './lafan2/'
    bvh_names = os.listdir(bvh_dir)
    bvh_names.sort()

    with open('db_index_info2.txt', 'w') as db_index_info:

        db_index = 0
        data = []
        for bvh_name in bvh_names:
            Joint.resize = 1
            bvh_path = os.path.join(bvh_dir, bvh_name)
            bvh = open(bvh_path, 'r')

            # db_index_info.txt 
            num_of_frames, FPS = parsing_bvh(bvh)
            info = ' '.join([str(db_index), bvh_name, str(state.line_index), str(FPS)+'\n'])
            db_index_info.write(info)

            db_index += num_of_frames - FPS 

            for i in range(0, len(state.frame_list) - FPS -1):
                
                state.curFrame = state.frame_list[i]
                state.futureFrames = []  
                for j in [int(FPS/3), int(FPS/3*2), int(FPS)]:    
                    if i + j < len(state.frame_list):
                        state.futureFrames.append(state.frame_list[i + j])
                set_joint_feature(state.joint_list[0], np.identity(4), None)
                feature_vector = Feature()
                set_feature_vector(feature_vector)
                state.feature_list.append(feature_vector)

                data.append(feature_vector.get_feature_list())

    DB = cKDTree(np.array(data))
    with open('tree_dump2.bin', 'wb') as dump_file:
        pickle.dump(DB, dump_file)

    with open('db_contents.txt', 'w') as f:
        for i in range(len(data)):
            f.write("#" + str(i) + str(data[i])+'\n')

    print("RESIZE ", Joint.resize)


if __name__ == "__main__":
    main()
