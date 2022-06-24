import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation as R

from Joint import Joint
from Feature import Feature
import utils
import MyWindow


class state:
    joint_list = []
    frame_list = []
    query_vector = Feature()
    bvh_past_position = np.array([])
    
    # The global position of the character on the window
    real_global_position = np.array([0., 0., 0.])


def parsing_bvh(bvh):

    frameTime = 0
    state.frame_list = []
    num_of_frames = 0

    line = bvh.readline().split()

    if line[0] == 'HIERARCHY':
        line = bvh.readline().split()
        if line[0] == 'ROOT':
            state.joint_list = []
            buildJoint(bvh, line[1])  # build ROOT and other joints
            Joint.resize = int(Joint.resize)

    line = bvh.readline().split()

    if line[0] == 'MOTION':
        line = bvh.readline().split()
        if line[0] == 'Frames:':
            num_of_frames = int(line[1])
        line = bvh.readline().split()
        if line[0] == 'Frame' and line[1] == 'Time:':
            frameTime = float(line[2])

        for i in range(num_of_frames):
            line = bvh.readline().split()
            line = list(map(float, line))
            state.frame_list.append(line)
        last = [0] * len(state.frame_list[0])   # what's this??
        state.frame_list.append(last)

    FPS = int(1 / frameTime)
    return FPS


def buildJoint(bvh, joint_name):

    line = bvh.readline().split()  # remove '{'
    newJoint = Joint(joint_name)

    # check if it's foot joint
    if "Foot" in joint_name:
        newJoint.set_is_foot(True)

    newJoint.set_index(len(state.joint_list))

    state.joint_list.append(newJoint)

    line = bvh.readline().split()
    if line[0] == 'OFFSET':
        offset = np.array(list(map(float, line[1:])), dtype='float32')
        if joint_name != "Hips" and np.sqrt(np.dot(offset, offset)) > Joint.resize:
            Joint.resize = np.sqrt(np.dot(offset, offset))
        newJoint.set_offset(offset)

    line = bvh.readline().split()
    if line[0] == 'CHANNELS':
        newJoint.set_channel(line[2:])

    while True:
        line = bvh.readline().split()
        if line[0] == 'JOINT':
            newJoint.append_child_joint(buildJoint(bvh, line[1]))

        elif line[0] == 'End' and line[1] == 'Site':
            line = bvh.readline().split()  # remove '{'
            line = bvh.readline().split()
            if line[0] == 'OFFSET':
                offset = np.array(list(map(float, line[1:])), dtype='float32')

                if joint_name != "Hips" and np.sqrt(np.dot(offset, offset)) > Joint.resize:
                    Joint.resize = np.sqrt(np.dot(offset, offset))

                newJoint.set_end_site(offset)
            line = bvh.readline().split()  # remove '}'

        elif line[0] == '}':
            return newJoint


# A function that draws a line (or capsule) connecting the parent point and my point
# the rootMatrix enters None when drawing a root joint
# the parentMatrix enters identity(4) when drawing a root joint

def drawJoint(parentMatrix, joint, rootMatrix=None):

    glPushMatrix()
    newMatrix = np.identity(4)
    cur_position = [0, 0, 0, 1]

    # get current joint's offset from parent joint
    curoffset = joint.get_offset() / Joint.resize

    # move transformation matrix's origin using offset data
    temp = np.identity(4)
    if len(joint.get_channel()) != 6:
        temp[:3, 3] = curoffset
    newMatrix = newMatrix @ temp

    # channel rotation
    # ROOT
    if len(joint.get_channel()) == 6:
        # if len(state.bvh_past_position) != 0: # Continuous motion playback received via the QnA function
        #     state.real_global_position += np.array(MyWindow.state.curFrame[:3]) - state.bvh_past_position
        # else:   # if QnA is newly called
        #     state.real_global_position[1] = MyWindow.state.curFrame[1]   
        # state.bvh_past_position = MyWindow.state.curFrame[:3]

        #ROOTPOSITION = np.array(MyWindow.state.curFrame[:3], dtype='float32')

        # ROOTPOSITION = np.array(state.real_global_position, dtype='float32')
        ROOTPOSITION = np.array(MyWindow.state.curFrame[:3])
        ROOTPOSITION /= Joint.resize
        # move root's transformation matrix's origin using translation data
        temp = np.identity(4)
        temp[:3, 3] = ROOTPOSITION
        newMatrix = newMatrix @ temp

        for i in range(3, 6):
            if joint.get_channel()[i].upper() == 'XROTATION':
                xr = MyWindow.state.curFrame[i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx
            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = MyWindow.state.curFrame[i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry
            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = MyWindow.state.curFrame[i]
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
                xr = MyWindow.state.curFrame[(index + 1) * 3 + i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr), 0],
                               [0, np.sin(xr), np.cos(xr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Rx

            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = MyWindow.state.curFrame[(index + 1) * 3 + i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr), 0.],
                               [0, 1, 0, 0],
                               [-np.sin(yr), 0, np.cos(yr), 0],
                               [0., 0., 0., 1.]])
                newMatrix = newMatrix @ Ry

            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = MyWindow.state.curFrame[(index + 1) * 3 + i]
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
    else:
        parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

    cur_position = global_position      # global position of this joint

    # Check if it's Root joint, otherwise update Joint class's data
    # velocity, rotation velocity update 

    if joint.get_is_root():
        rootMatrix = joint.get_transform_matrix()

    else:
        # get root local position and root local velocity
        new_root_local_position = (rootMatrix.T @ global_position)[:3]  # local to root joint
        past_root_local_position = joint.get_root_local_position()  # local to root joint
        root_local_velocity = (new_root_local_position - past_root_local_position) * 30   # 30 is FPS of LaFAN1

        # get root local rotation and root local angular velocity
        new_root_local_rotation_matrix = (rootMatrix.T @ transform_matrix)[:3, :3]
        r = R.from_matrix(new_root_local_rotation_matrix)
        new_root_local_rotation = np.array(r.as_quat())

        # set joint class's value
        joint.set_global_position(global_position[:3])
        joint.set_root_local_velocity(root_local_velocity)
        joint.set_root_local_position(new_root_local_position[:3])
        joint.set_root_local_rotation(new_root_local_rotation)

    v = cur_position - parent_position
    box_length = utils.l2norm(v)
    v = utils.normalized(v)
    rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
    check = np.dot(np.array([0, 1, 0]), v[:3])
    # under 90
    if check >= 0:
        rotate_angle = np.arcsin(utils.l2norm(rotation_vector))
        rotation_vector = utils.normalized(rotation_vector) * rotate_angle
    # over 90
    else:
        rotate_angle = np.arcsin(utils.l2norm(rotation_vector)) + np.pi
        rotate_angle *= -1

        # not 180
        if utils.l2norm(rotation_vector) != 0:
            rotation_vector = utils.normalized(rotation_vector) * rotate_angle
        # if 180, rotate_vector becomes (0, 0, 0)
        else:
            rotation_vector = np.array([0., 0., 1.]) * rotate_angle

    rotation_matrix = np.identity(4)
    rotation_matrix[:3, :3] = utils.exp(rotation_vector[:3])

    glPushMatrix()
    glTranslatef(parent_position[0], parent_position[1], parent_position[2])
    glMultMatrixf(rotation_matrix.T)
    glTranslatef(0., box_length / 2, 0.)
    glScalef(.05, box_length, .05)
    # drawCapsule
    quadric = gluNewQuadric()
    gluSphere(quadric, 0.5, 20, 20)
    glPopMatrix()

    # draw end effector
    if joint.get_end_site() is not None:
        end_offset = joint.get_end_site() / joint.resize
        endMatrix = np.identity(4)
        endMatrix[:3, 3] = end_offset
        end_position = parentMatrix @ newMatrix @ endMatrix @ np.array([0., 0., 0., 1.])

        v = end_position - cur_position
        box_length = utils.l2norm(v)
        v = utils.normalized(v)
        rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
        check = np.dot(np.array([0, 1, 0]), v[:3])
        if check >= 0:
            rotate_angle = np.arcsin(utils.l2norm(rotation_vector))
        else:
            rotate_angle = np.arcsin(utils.l2norm(rotation_vector)) + np.pi
            rotate_angle *= -1
        rotation_vector = utils.normalized(rotation_vector) * rotate_angle
        rotation_matrix = np.identity(4)
        rotation_matrix[:3, :3] = utils.exp(rotation_vector[:3])

        glPushMatrix()
        glTranslatef(cur_position[0], cur_position[1], cur_position[2])
        glMultMatrixf(rotation_matrix.T)
        glTranslatef(0., box_length / 2, 0.)
        glScalef(.05, box_length, .05)
        # drawCapsule
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.5, 20, 20)
        glPopMatrix()

    # draw child joints
    else:
        for j in joint.get_child():
            drawJoint(joint.get_transform_matrix(), j, rootMatrix)

    glPopMatrix()


def set_query_vector(key_input = None):

    two_foot_position = []
    two_foot_velocity = []
    hip_velocity = []

    # future facing direction setting
    default_facing_direction = np.array([1., 0., 0.])
    rotation_current = R.from_euler('zyx', MyWindow.state.curFrame[3:6], degrees=True)
    rotation_current = rotation_current.as_matrix()

    if key_input is not None:

        if key_input == "UP":
            yr = np.radians(135)
        elif key_input == "DOWN":
            yr = np.radians(-45)
        elif key_input == "RIGHT":
            yr = np.radians(45)
        elif key_input == "LEFT":
            yr = np.radians(-135)

        rotation_future = np.array([[np.cos(yr), 0, np.sin(yr)],
                            [0, 1, 0],
                            [-np.sin(yr), 0, np.cos(yr)]])

    else:   # key_input == None
        rotation_future = rotation_current  # no change, keep going

    future3Direction = (rotation_current.T @ rotation_future) @ default_facing_direction
    future2Direction = future3Direction[0::2]
    future2Directions = [future2Direction, future2Direction, future2Direction]  # future 10, 20, 30 direction

    # future trajectory setting
    default_future_trajectory = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])     # future 10, 20, 30 position
    futurePosition = future3Direction * default_future_trajectory
    futurePosition = futurePosition[:, 0::2]

    for joint in state.joint_list:
        if joint.get_is_root():
            hip_velocity.append(joint.get_root_local_velocity())
        elif joint.get_is_foot():
            two_foot_position.append(joint.get_root_local_position())
            two_foot_velocity.append(joint.get_root_local_velocity())

    # state.query_vector.set_future_position(np.array(futurePosition).reshape(6, ))
    # state.query_vector.set_future_direction(np.array(future2Directions).reshape(6, ))
    
    # test future info setting
    state.query_vector.set_future_position(np.zeros_like(futurePosition).reshape(6, ))
    state.query_vector.set_future_direction(np.zeros_like(future2Directions).reshape(6, ))

    state.query_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
    state.query_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
    state.query_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))

    return state.query_vector.get_feature_list()


def reset_bvh_past_postion():
    state.bvh_past_position = np.array([])
