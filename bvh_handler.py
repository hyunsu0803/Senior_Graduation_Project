import numpy as np
from Joint import Joint
from utils import l2norm, normalized, exp

from OpenGL.GL import *
from OpenGL.GLU import *

joint_list = []
num_of_frames = 0
frame_list = []


def parsing_bvh(bvh):
    global num_of_frames, frame_list
    frameTime = 0
    frame_list = []

    line = bvh.readline().split()

    if line[0] == 'HIERARCHY':
        line = bvh.readline().split()
        if line[0] == 'ROOT':
            buildJoint(bvh, line[1])  # build ROOT and other joints

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
            frame_list.append(line)
        last = [0] * len(frame_list[0])
        frame_list.append(last)

    FPS = int(1 / frameTime)
    return FPS


def buildJoint(bvh, joint_name):
    global joint_list
    joint_list = []

    line = bvh.readline().split()  # remove '{'
    newJoint = Joint(joint_name)
    newJoint.set_index(len(joint_list))
    joint_list.append(newJoint)

    line = bvh.readline().split()
    if line[0] == 'OFFSET':
        offset = np.array(list(map(float, line[1:])), dtype='float32')
        if np.sqrt(np.dot(offset, offset)) > Joint.resize:
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
                if np.sqrt(np.dot(offset, offset)) > Joint.resize:
                    Joint.resize = np.sqrt(np.dot(offset, offset))
                newJoint.set_end_site(offset)
            line = bvh.readline().split()  # remove '}'

        elif line[0] == '}':
            return newJoint


def drawJoint(parentMatrix, joint):
    from MyWindow import curFrame

    glPushMatrix()
    newMatrix = np.identity(4)

    parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

    # offset translation
    curoffset = joint.get_offset() / Joint.resize

    temp = np.identity(4)
    temp[:3, 3] = curoffset
    newMatrix = newMatrix @ temp

    cur_position = parentMatrix @ newMatrix @ np.array([0., 0., 0., 1.])

    # channel rotation
    # ROOT
    if len(joint.get_channel()) == 6:
        ROOTPOSITION = np.array(curFrame[:3], dtype='float32')

        ROOTPOSITION /= Joint.resize
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

    v = cur_position - parent_position
    box_length = l2norm(v)
    v = normalized(v)
    rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
    check = np.dot(np.array([0, 1, 0]), v[:3])
    # under 90
    if check >= 0:
        rotate_angle = np.arcsin(l2norm(rotation_vector))
        rotation_vector = normalized(rotation_vector) * rotate_angle
    # over 90
    else:
        rotate_angle = np.arcsin(l2norm(rotation_vector)) + np.pi
        rotate_angle *= -1

        # not 180
        if l2norm(rotation_vector) != 0:
            rotation_vector = normalized(rotation_vector) * rotate_angle
        # if 180, rotate_vector becomes (0, 0, 0)
        else:
            rotation_vector = np.array([0., 0., 1.]) * rotate_angle

    rotation_matrix = np.identity(4)
    rotation_matrix[:3, :3] = exp(rotation_vector[:3])

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
        box_length = l2norm(v)
        v = normalized(v)
        rotation_vector = np.cross(np.array([0, 1, 0]), v[:3])
        check = np.dot(np.array([0, 1, 0]), v[:3])
        if check >= 0:
            rotate_angle = np.arcsin(l2norm(rotation_vector))
        else:
            rotate_angle = np.arcsin(l2norm(rotation_vector)) + np.pi
            rotate_angle *= -1
        rotation_vector = normalized(rotation_vector) * rotate_angle
        rotation_matrix = np.identity(4)
        rotation_matrix[:3, :3] = exp(rotation_vector[:3])

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
            drawJoint(joint.get_transform_matrix(), j)

    glPopMatrix()
