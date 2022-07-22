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
        if len(state.bvh_past_position) != 0: # Continuous motion playback received via the QnA function
            print("I'm old!!")
            print("curFrame data position", MyWindow.state.curFrame[:3])
            print("pastFrame data position", state.bvh_past_position)
            state.real_global_position += (np.array(MyWindow.state.curFrame[:3]) - np.array(state.bvh_past_position))
            
        else:   # if QnA is newly called
            print("I'm new!!!")
            print("curFrame data position", MyWindow.state.curFrame[:3])
            state.real_global_position[1] = MyWindow.state.curFrame[1]
            print("new query!!")
        state.bvh_past_position = MyWindow.state.curFrame[:3]

        ROOTPOSITION = np.array(state.real_global_position, dtype='float32')
        print(ROOTPOSITION)
        ROOTPOSITION /= Joint.resize
        #print(ROOTPOSITION)

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
        print("resized position", global_position)
    else:
        parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

    cur_position = global_position      # global position of this joint

    # Check if it's Root joint, otherwise update Joint class's data
    # velocity, rotation velocity update 

    if joint.get_is_root():
        rootMatrix = joint.get_transform_matrix()
        global_velocity = (global_position[:3]- joint.get_global_position()) * 30
        joint.set_global_position(global_position[:3])
        joint.set_global_velocity(global_velocity)
        drawLocalFrame(rootMatrix)
        

    else:
        # get root local position and root local velocity
        new_root_local_position = (np.linalg.inv(rootMatrix) @ global_position)[:3]  # local to root joint
        past_root_local_position = joint.get_root_local_position()  # local to root joint
        root_local_velocity = (new_root_local_position - past_root_local_position) * 30   # 30 is FPS of LaFAN1

        # get root local rotation and root local angular velocity
        new_root_local_rotation_matrix = (np.linalg.inv(rootMatrix) @ transform_matrix)[:3, :3]
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

    for joint in state.joint_list:
        if joint.get_is_root():
            hip_velocity.append(joint.get_root_local_velocity())
        elif joint.get_is_foot():
            two_foot_position.append(joint.get_root_local_position())
            two_foot_velocity.append(joint.get_root_local_velocity())
    

    # future direction setting
    future_direction = None
    if key_input == None:
        future_direction = np.array([0., 1., 0.])
    elif key_input == "LEFT":
        global_future_direction = np.array([1., 0., 0.])
        future_direction = joint.get_transform_matrix()[:3, :3].T @ global_future_direction
    elif key_input == "RIGHT":
        global_future_direction = np.array([-1., 0., 0.])
        future_direction = joint.get_transform_matrix()[:3, :3].T @ global_future_direction
    elif key_input == "UP":
        global_future_direction = np.array([0., 0., 1.])
        future_direction = joint.get_transform_matrix()[:3, :3].T @ global_future_direction
    elif key_input == "DOWN":
        global_future_direction = np.array([0., 0., -1.])
        future_direction = joint.get_transform_matrix()[:3, :3].T @ global_future_direction

    local_3Ddirection_future = np.array([future_direction, future_direction, future_direction])
    local_2Ddirection_future = local_3Ddirection_future[:, 1:]

    # future position setting
    abs_global_velocity = np.linalg.norm(state.joint_list[0].get_global_velocity())/3
    local_3Dposition_future = np.zeros((3, 3))

    for i in range(3):
        local_3Dposition_future[i] = future_direction * (abs_global_velocity * i)
    local_2Dposition_future = local_3Dposition_future[:, 1:]

    # global direction setting
    global_direction = state.joint_list[0].getGlobalDirection()
    global_3Ddirection_future = np.array([global_direction, global_direction, global_direction])

    # global position setting
    global_3Dposition_future = np.zeros((3, 3))

    for i in range(3):
        global_3Dposition_future[i] = global_direction * (abs_global_velocity * i)

    state.query_vector.set_global_future_direction(global_3Ddirection_future)
    state.query_vector.set_global_future_position(global_3Dposition_future)
    state.query_vector.set_future_position(np.array(local_2Dposition_future).reshape(6, ))
    state.query_vector.set_future_direction(np.array(local_2Ddirection_future).reshape(6, ))
    state.query_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
    state.query_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
    state.query_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))

    return state.query_vector.get_feature_list()

 
def draw_future_info():
    # global 3d info
    future_position = state.query_vector.get_global_future_position().reshape(3, 3)
    future_direction = state.query_vector.get_global_future_direction().reshape(3, 3)

    future_position[:, 1] = 0.
    future_direction[:, 1] = 0.

    glPointSize(20.)
    glBegin(GL_POINTS)
    glVertex3fv(future_position[0])
    glVertex3fv(future_position[1])
    glVertex3fv(future_position[2])
    glEnd()

    glLineWidth(5.)
    glBegin(GL_LINE_STRIP)
    glVertex3fv(future_position[0])
    glVertex3fv(future_position[0]+future_direction[0])

    glVertex3fv(future_position[1])
    glVertex3fv(future_position[1]+future_direction[1])

    glVertex3fv(future_position[2])
    glVertex3fv(future_position[2]+future_direction[2])
    glEnd()



def reset_bvh_past_postion():
    state.bvh_past_position = np.array([])

def drawLocalFrame(M):
    glPushMatrix()
    glTranslatef(M[0][3], M[1][3], M[2][3])
    rotationMatrix = np.identity(4)
    rotationMatrix[:3, :3] = M[:3, :3]
    glMultMatrixf(rotationMatrix.T)
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([0., 0., 0.]))
    glVertex3fv(np.array([1., 0., 0.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0., 0., 0.]))
    glVertex3fv(np.array([0., 3., 0.]))
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0., 0., 0]))
    glVertex3fv(np.array([0., 0., 5.]))
    glEnd()
    glPopMatrix()    
