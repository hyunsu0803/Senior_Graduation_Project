from telnetlib import theNULL
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
    bvh_past_orientation = np.array([])
    target_orientation = np.array([1., 0., 0.])
    
    # The global position of the character on the window
    real_global_position = np.array([0., 0., 0.])
    TposeX = [0, 1, 0]
    TposeY = [1, 0, 0]
    TposeZ = [0, 0, 1]
    real_global_orientation = np.array([TposeX, TposeY, TposeZ]).T

    mean_array = np.array([ 1.41277287e-02,  4.27515134e-01,  3.89398700e-02,  8.42000393e-01,
                            7.17285028e-02,  1.20716830e+00, -8.23741644e-02,  1.84959764e-02,
                            -8.22155663e-02,  1.82737104e-02, -8.15859744e-02,  1.79583760e-02,
                            2.96535476e-01,  4.09111197e-01, -5.73196245e-02, -3.08930586e-01,
                            4.04232245e-01, -6.19720699e-02,  1.28611388e-03,  1.24453725e-03,
                            -4.38613943e-04, -1.33293606e-03,  1.24974156e-03, -1.89162694e-04,
                            2.22365800e-19,  1.00846789e-02,  2.17101458e-19])
    std_array = np.array([3.67570835e-01, 7.05172893e-01, 7.44494015e-01, 1.36265022e+00,
                            1.16910184e+00, 1.95419930e+00, 7.56352291e-01, 6.48693771e-01,
                            7.52681095e-01, 6.52976142e-01, 7.51695391e-01, 6.54198184e-01,
                            3.98546797e-01, 4.81824252e-01, 5.52025651e-01, 3.83578971e-01,
                            4.77795089e-01, 5.42710984e-01, 1.74015595e+00, 1.33964037e+00,
                            2.69185055e+00, 1.76546581e+00, 1.36998779e+00, 2.68857973e+00,
                            3.24655240e-14, 1.16281390e+00, 2.67810824e-14])


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

def drawJoint(parentMatrix, joint, characterMatrix=None):

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
        bvh_current_orientation = np.identity(3)
        bvh_to_real_rotation = None

        for i in range(3, 6):
            if joint.get_channel()[i].upper() == 'XROTATION':
                xr = MyWindow.state.curFrame[i]
                xr = np.radians(xr)
                Rx = np.array([[1., 0., 0.],
                               [0, np.cos(xr), -np.sin(xr)],
                               [0, np.sin(xr), np.cos(xr)]])
                bvh_current_orientation = bvh_current_orientation @ Rx
            elif joint.get_channel()[i].upper() == 'YROTATION':
                yr = MyWindow.state.curFrame[i]
                yr = np.radians(yr)
                Ry = np.array([[np.cos(yr), 0, np.sin(yr)],
                               [0, 1, 0],
                               [-np.sin(yr), 0, np.cos(yr)]])
                bvh_current_orientation = bvh_current_orientation @ Ry
            elif joint.get_channel()[i].upper() == 'ZROTATION':
                zr = MyWindow.state.curFrame[i]
                zr = np.radians(zr)
                Rz = np.array([[np.cos(zr), -np.sin(zr), 0],
                               [np.sin(zr), np.cos(zr), 0],
                               [0, 0, 1]])
                bvh_current_orientation = bvh_current_orientation @ Rz

        # calculate real global orientation
        if len(state.bvh_past_orientation) != 0: #Continuous motion playback received via the QnA function
            bvh_past_to_current_rotation =  bvh_current_orientation @ state.bvh_past_orientation.T   # about global frame
            state.real_global_orientation = bvh_past_to_current_rotation @ state.real_global_orientation
            bvh_to_real_rotation = state.real_global_orientation @ bvh_current_orientation.T # about global frame

        else:   # if QnA is newly called
            bvh_current_orientation_direction = bvh_current_orientation[:3, 1].copy()
            bvh_current_orientation_direction[1] = 0
            bvh_current_orientation_direction = utils.normalized(bvh_current_orientation_direction)

            real_global_orientation_direction = state.real_global_orientation[:3, 1].copy()
            real_global_orientation_direction[1] = 0
            real_global_orientation_direction = utils.normalized(real_global_orientation_direction)

            th = np.arccos(np.dot(bvh_current_orientation_direction, real_global_orientation_direction))
            crossing = np.cross(bvh_current_orientation_direction, real_global_orientation_direction)

            if crossing[1] < 0:
                th *= -1
            
            bvh_to_real_rotation_yrotation = np.array([[np.cos(th), 0, np.sin(th)],
                                             [0, 1, 0],
                                             [-np.sin(th), 0, np.cos(th)]]) # about global frame

            state.real_global_orientation = bvh_to_real_rotation_yrotation @ bvh_current_orientation

        state.bvh_past_orientation = bvh_current_orientation

        # calculate real global position
        if len(state.bvh_past_position) != 0: # Continuous motion playback received via the QnA function
            movement_vector = (np.array(MyWindow.state.curFrame[:3]) - np.array(state.bvh_past_position))
            state.real_global_position += bvh_to_real_rotation @ movement_vector    
        else:   # if QnA is newly called
            state.real_global_position[1] = MyWindow.state.curFrame[1]
            
        state.bvh_past_position = MyWindow.state.curFrame[:3]

        ROOTPOSITION = np.array(state.real_global_position, dtype='float32')
        ROOTPOSITION /= Joint.resize

        # move root's transformation matrix's origin using translation data
        newMatrix[:3, 3] = ROOTPOSITION
        newMatrix[:3, :3] = state.real_global_orientation

        #print("@@@@@", state.real_global_orientation.T @ np.array([0.,0., 1]))

        joint.set_transform_matrix(parentMatrix @ newMatrix)
        transform_matrix = joint.get_transform_matrix()
    
        global_position = transform_matrix @ np.array([0., 0., 0., 1.])

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
        characterMatrix = joint.getCharacterLocalFrame().copy()
        global_velocity = (global_position[:3]- joint.get_global_position()) * 30
        joint.set_global_position(global_position[:3])
        joint.set_global_velocity(global_velocity)
        drawLocalFrame(characterMatrix)
        

    # get root local position and root local velocity
    new_character_local_position = (np.linalg.inv(characterMatrix) @ global_position)[:3]  # local to root joint
    past_character_local_position = joint.get_character_local_position()  # local to root joint
    character_local_velocity = (new_character_local_position - past_character_local_position) * 30   # 30 is FPS of LaFAN1)

    # set joint class's value
    joint.set_global_position(global_position[:3])
    joint.set_character_local_velocity(character_local_velocity)
    joint.set_character_local_position(new_character_local_position[:3])

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
            drawJoint(joint.get_transform_matrix(), j, characterMatrix)

    glPopMatrix()


def set_query_vector(key_input = None):

    two_foot_position = []
    two_foot_velocity = []
    hip_velocity = []

    for joint in state.joint_list:
        if joint.get_is_root():
            hip_velocity.append(joint.get_character_local_velocity())
        elif joint.get_is_foot():
            two_foot_position.append(joint.get_character_local_position())
            two_foot_velocity.append(joint.get_character_local_velocity())
    

    # future direction setting
    future_direction = None
    global_future_direction = None

    if key_input == None:
        global_future_direction = state.target_orientation
        future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
    elif key_input == "LEFT":
        state.target_orientation = np.array([1., 0., 0.])
        global_future_direction = state.target_orientation
        future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
    elif key_input == "RIGHT":
        state.target_orientation = np.array([-1., 0., 0.])
        global_future_direction = state.target_orientation
        future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
    elif key_input == "UP":
        state.target_orientation = np.array([0., 0., 1.])
        global_future_direction = state.target_orientation
        future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction
    elif key_input == "DOWN":
        state.target_orientation = np.array([0., 0., -1.])
        global_future_direction = state.target_orientation
        future_direction = state.joint_list[0].getCharacterLocalFrame()[:3, :3].T @ global_future_direction


    local_3Ddirection_future = np.array([future_direction, future_direction, future_direction])
    local_2Ddirection_future = local_3Ddirection_future[:, 0::2]

    # future position setting
    #abs_global_velocity = np.linalg.norm(state.joint_list[0].get_global_velocity())/3
    abs_global_velocity = 1.3
    local_3Dposition_future = np.zeros((3, 3))

    for i in range(3):
        local_3Dposition_future[i] = future_direction * (abs_global_velocity * (i+1))
    print("case1) !!!!!!!!!!")
    print(local_3Dposition_future)

    local_2Dposition_future = local_3Dposition_future[:, 0::2]

    # global direction setting
    global_3Ddirection_future = np.array([global_future_direction, global_future_direction, global_future_direction])

    # global position setting
    global_3Dposition_future = np.zeros((3, 3))

    for i in range(3):
        global_3Dposition_future[i] = state.real_global_position/Joint.resize + global_future_direction * (abs_global_velocity * (i+1))
    global_3Dposition_future[:, 1] = 0
    print("global position")
    print(global_3Dposition_future)
    print("case2) !!!!!!!!!!")
    for i in range(3):
        temp = np.array([0., 0., 0., 1])
        temp[:3] = global_3Dposition_future[i]
        print(np.linalg.inv(state.joint_list[0].getCharacterLocalFrame()) @ temp)
        

    state.query_vector.set_global_future_position(global_3Dposition_future)
    state.query_vector.set_global_future_direction(global_3Ddirection_future)

    state.query_vector.set_future_position(np.array(local_2Dposition_future).reshape(6, ))
    state.query_vector.set_future_direction(np.array(local_2Ddirection_future).reshape(6, ))
    state.query_vector.set_foot_position(np.array(two_foot_position).reshape(6, ))
    state.query_vector.set_foot_velocity(np.array(two_foot_velocity).reshape(6, ))
    state.query_vector.set_hip_velocity(np.array(hip_velocity).reshape(3, ))
    feature_vector = state.query_vector.get_feature_list().copy()

    # normalization
    for i in range(0, 27):
        feature_vector[i] = (feature_vector[i] - state.mean_array[i]) / state.std_array[i]

    for i in range(0, 12):
        feature_vector[i] = feature_vector[i] * 2

    return feature_vector

 
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

def reset_bvh_past_orientation():
    state.bvh_past_orientation = np.array([])

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
