import numpy as np
from Joint import Joint
from Feature import Feature
from utils import l2norm, normalized, exp

from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation as R

joint_list = []
num_of_frames = 0
frame_list = []
feature_list = []
timeStep = 0.2
line_index = 0
curFrame = []
futureFrames = [None, None, None]
futurePosition = [[None], [None], [None]]
futureDirection = [[None], [None], [None]]


def parsing_bvh(bvh):
    global num_of_frames, frame_list, joint_list, line_index
    frameTime = 0
    frame_list = []
    line_index = 0

    line = bvh.readline().split()
    line_index += 1

    if line[0] == 'HIERARCHY':
        line = bvh.readline().split()
        line_index += 1
        if line[0] == 'ROOT':
            joint_list = []
            buildJoint(bvh, line[1])  # build ROOT and other joints

    line = bvh.readline().split()
    line_index += 1

    if line[0] == 'MOTION':
        line = bvh.readline().split()
        line_index += 1
        if line[0] == 'Frames:':
            num_of_frames = int(line[1])
        line = bvh.readline().split()
        line_index += 1

        if line[0] == 'Frame' and line[1] == 'Time:':
            frameTime = float(line[2])

        for i in range(num_of_frames):
            line = bvh.readline().split()
            line = list(map(float, line))
            frame_list.append(line)
        last = [0] * len(frame_list[0])
        frame_list.append(last)

    FPS = int(1 / frameTime)
    return FPS


def buildJoint(bvh, joint_name):
    global joint_list, line_index
    # joint_list = []

    line = bvh.readline().split()  # remove '{'
    line_index += 1
    newJoint = Joint(joint_name)

    # check if it's foot joint
    # 이건 사용하는 data set에 따라 달라질 수 있음(관절 이름이 달라질 수 있으니까)
    if "Foot" in joint_name:
        newJoint.set_is_foot(True)

    newJoint.set_index(len(joint_list))

    joint_list.append(newJoint)

    line = bvh.readline().split()
    line_index += 1
    if line[0] == 'OFFSET':
        offset = np.array(list(map(float, line[1:])), dtype='float32')
        if np.sqrt(np.dot(offset, offset)) > Joint.resize:
            Joint.resize = np.sqrt(np.dot(offset, offset))
        newJoint.set_offset(offset)

    line = bvh.readline().split()
    line_index += 1
    if line[0] == 'CHANNELS':
        newJoint.set_channel(line[2:])

    while True:
        line = bvh.readline().split()
        line_index += 1
        if line[0] == 'JOINT':
            newJoint.append_child_joint(buildJoint(bvh, line[1]))

        elif line[0] == 'End' and line[1] == 'Site':
            line = bvh.readline().split()  # remove '{'
            line_index += 1
            line = bvh.readline().split()
            line_index += 1
            if line[0] == 'OFFSET':
                offset = np.array(list(map(float, line[1:])), dtype='float32')
                if np.sqrt(np.dot(offset, offset)) > Joint.resize:
                    Joint.resize = np.sqrt(np.dot(offset, offset))
                newJoint.set_end_site(offset)
            line = bvh.readline().split()  # remove '}'
            line_index += 1

        elif line[0] == '}':
            return newJoint


def set_joint_feature(joint, parentMatrix, rootMatrix=None):
    global curFrame, timeStep, futurePosition, futureDirection
    newMatrix = np.identity(4)
    cur_position = [0, 0, 0, 1]

    # get current joint's offset from parent joint
    curoffset = joint.get_offset() / Joint.resize

    temp = np.identity(4)
    temp[:3, 3] = curoffset
    newMatrix = newMatrix @ temp

    # channel rotation
    # ROOT
    if len(joint.get_channel()) == 6:
        ROOTPOSITION = np.array(curFrame[:3], dtype='float32')
        ROOTPOSITION /= Joint.resize

        # move root's transformation matrix's origin using translation data
        temp = np.identity(4)
        temp[:3, 3] = ROOTPOSITION
        newMatrix = newMatrix @ temp

        for i in range(3, 6):
            if joint.get_channel()[i].upper() == 'XROTATION':
                xr = curFrame[i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx
            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = curFrame[i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry
            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = curFrame[i]
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
                xr = curFrame[(index + 1) * 3 + i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx

            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = curFrame[(index + 1) * 3 + i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry

            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = curFrame[(index + 1) * 3 + i]
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
    if joint.get_is_root() is not None:
        parent_position = global_position
        futurePosition = [None, None, None]
        futureDirection = [None, None, None]

        for i in range(0, 3):
            temp = np.array([0, 0, 0, 1])
            temp[:3] = futureFrames[i][:3]
            futurePosition[i] = np.linalg.inv(transform_matrix) @ temp

            futurePosition[i] = futurePosition[i][0::2]
            # 나중에 zxy 아닐 수도 있으니 바꾸쇼~!

            # 만약 나중에 안된다면 이거때문일수도 있음! 자신이 없어~~~~
            default_facing_direction = np.array([1., 0., 0.])
            rotation_current = R.from_euler('zxy', curFrame[3:6], degrees=True)
            rotation_future = R.from_euler('zxy', futureFrames[i][3:6], degrees=True)
            rotation_current = np.array(rotation_current.as_matrix())
            rotation_future = np.array(rotation_future.as_matrix())
            futureDirection[i] = R.from_matrix(rotation_current.T @ rotation_future).as_matrix() @ default_facing_direction
            futureDirection[i] = futureDirection[i][0::2]

    else:
        parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

    cur_position = global_position

    # Check if it's Root joint, otherwise update Joint class's data
    # velocity, rotation velocity update 시키기

    if joint.get_is_root() is not None:
        rootMatrix = joint.get_transform_matrix()

    else:
        # get root local position and root local velocity
        new_root_local_position = (rootMatrix.T @ global_position)[:3]  # local to root joint
        past_root_local_position = joint.get_root_local_position()  # local to root joint
        root_local_velocity = ((new_root_local_position - past_root_local_position) / timeStep)

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

    print(joint.joint_name)
    if joint.get_is_root() is not None:
        print("is root!!!")
    print("global position: ")
    print(joint.get_global_position())
    print("root local position")
    print(joint.get_root_local_position())
    print("root local velocity")
    print(joint.get_root_local_velocity())
    print("root local rotation")
    print(joint.get_root_local_rotation())
    print()
    print()

    if joint.get_end_site() is None:
        for j in joint.get_child():
            set_joint_feature(j, joint.get_transform_matrix(), rootMatrix)


def set_feature_vector(feature_vector):
    global joint_list

    two_foot_position = []
    two_foot_velocity = []
    hip_velocity = []

    for joint in joint_list:
        if joint.get_is_root() is not None:
            hip_velocity.append(joint.get_root_local_velocity())
        elif joint.get_is_foot() is not None:
            two_foot_position.append(joint.get_root_local_position())
            two_foot_velocity.append(joint.get_root_local_velocity())

    feature_vector.set_future_position(np.array(futurePosition).reshape(6, ))
    feature_vector.set_future_direction(np.array(futureDirection).reshape(6, ))
    feature_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
    feature_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
    feature_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))


def main():
    global curFrame, joint_list, feature_list, futureFrames
    curFrame = []
    Joint.resize = 1

    paths = []
    paths.append('sample-walk.bvh')

    with open(paths[0], 'r') as file:
        FPS = parsing_bvh(file)
        file_name = (paths[0].split('/'))[-1].strip(".bvh")

    f = open("features.txt", 'w')
    f.write(paths[0] + '\n')
    print(paths[0] + '\n')

    for i in range(0, len(frame_list) - 61):
        curFrame = frame_list[i]
        futureFrames = []
        for j in [20, 40, 60]:
            if j < len(frame_list):
                futureFrames.append(frame_list[i + j])
        set_joint_feature(joint_list[0], np.identity(4), None)
        feature_vector = Feature()
        set_feature_vector(feature_vector)
        feature_list.append(feature_vector)
        data = str(line_index + i) + " "
        data += feature_vector.get_feature_string()
        print(data)
        data += "\n"
        f.write(data)

    f.close()


if __name__ == "__main__":
    main()
