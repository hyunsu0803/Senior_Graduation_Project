from sklearn.feature_selection import SelectFdr
from Joint import Joint
import numpy as np
import utils
from OpenGL.GL import *
from OpenGL.GLU import *

class Character:

    def __init__(self):
        self.joint_list = []
        self.resize = 1
        self.setTpose()

    # only used in class
    def setTpose(self):
        # paths = './lafan1/fallAndGetUp1_subject1.bvh'
        paths = '/Users/jangbogyeong/my-awesome-project/lafan1/aiming1_subject1.bvh'

        with open(paths, 'r') as bvh:
            line = bvh.readline().split()

            if line[0] == 'HIERARCHY':
                line = bvh.readline().split()
                if line[0] == 'ROOT':
                    self.joint_list = []
                    self.buildCharacter(bvh, line[1])  # build ROOT and other joints
                    self.resize = int(self.resize)

    def drawCharacter(self):
        print(len(self.joint_list))
        self.drawJoint(np.identity(4), self.joint_list[0])


    def getCharacterRootJoint(self):

        root_joint = None
        for joint in self.joint_list:
            if joint.get_is_root():
                root_joint = joint
                break
        
        return root_joint

    def getCharacterTwoFootJoint(self):
        two_foot_joint_list = []
        for joint in self.joint_list:
            if joint.get_is_foot():
                two_foot_joint_list.append(joint)

        return two_foot_joint_list


    def getCharacterLocalFrame(self):
        rootJoint = self.getCharacterRootJoint()

        M = rootJoint.get_transform_matrix().copy()

        # origin projection 
        newOrigin = M[:3, 3]
        newOrigin[1] = 0

        newDirection = M[:3, 1]
        newDirection[1] = 0.

        newZaxis=  utils.normalized(newDirection)

        newYaxis = np.array([0., 1., 0.])
        newXaxis = utils.normalized(np.cross(newYaxis, newZaxis))

        CharacterMatrix = np.identity(4)
        CharacterMatrix[:3, 0] = newXaxis
        CharacterMatrix[:3, 1] = newYaxis
        CharacterMatrix[:3, 2] = newZaxis
        CharacterMatrix[:3, 3] = newOrigin

        return CharacterMatrix

    def getGlobalDirection(self):
        # find character global diredction
        M = self.getCharacterLocalFrame().copy()

        # length 1, projected on the ground
        return M[:3, 2]


    # used only inside class
    def buildCharacter(self, bvh, joint_name):

        line = bvh.readline().split()  # remove '{'
        newJoint = Joint(joint_name)

        # check if it's foot joint
        if "Foot" in joint_name:
            newJoint.set_is_foot(True)

        newJoint.set_index(len(self.joint_list))

        self.joint_list.append(newJoint)

        line = bvh.readline().split()
        if line[0] == 'OFFSET':
            offset = np.array(list(map(float, line[1:])), dtype='float32')
            if joint_name != "Hips" and np.sqrt(np.dot(offset, offset)) > self.resize:
                self.resize = np.sqrt(np.dot(offset, offset))
            newJoint.set_offset(offset)

        line = bvh.readline().split()
        if line[0] == 'CHANNELS':
            newJoint.set_channel(line[2:])

        while True:
            line = bvh.readline().split()
            if line[0] == 'JOINT':
                newJoint.append_child_joint(self.buildCharacter(bvh, line[1]))

            elif line[0] == 'End' and line[1] == 'Site':
                line = bvh.readline().split()  # remove '{'
                line = bvh.readline().split()
                if line[0] == 'OFFSET':
                    offset = np.array(list(map(float, line[1:])), dtype='float32')

                    if joint_name != "Hips" and np.sqrt(np.dot(offset, offset)) > self.resize:
                        self.resize = np.sqrt(np.dot(offset, offset))

                    newJoint.set_end_site(offset)
                line = bvh.readline().split()  # remove '}'

            elif line[0] == '}':
                return newJoint

    

    # used only inside class
    def drawJoint(self, parentMatrix, joint):
        
        transform_matrix = joint.get_transform_matrix()
        global_position = transform_matrix @ np.array([0., 0., 0., 1.])

        # set parent's global position (if it is root joint, parent_position is current_position)
        if joint.get_is_root():
            parent_position = global_position
        else:
            parent_position = parentMatrix @ np.array([0., 0., 0., 1.])

        cur_position = global_position      # global position of this joint
        
        v = cur_position/self.resize - parent_position/self.resize
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
        glColor3ub(255, 255, 255)
        glTranslatef(parent_position[0]/self.resize, parent_position[1]/self.resize, parent_position[2]/self.resize)
        glMultMatrixf(rotation_matrix.T)

        glTranslatef(0., box_length / 2, 0.)
        glScalef(.05, box_length, .05)
        # drawCapsule
        quadric = gluNewQuadric()
        gluSphere(quadric, 0.5, 20, 20)
        glPopMatrix()

        # draw end effector
        if joint.get_end_site() is not None:
            end_offset = joint.get_end_site()
            endMatrix = np.identity(4)
            endMatrix[:3, 3] = end_offset
            end_position = joint.get_transform_matrix() @ endMatrix @ np.array([0., 0., 0., 1.])

            v = end_position/self.resize - cur_position/self.resize
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
            glTranslatef(cur_position[0]/self.resize, cur_position[1]/self.resize, cur_position[2]/self.resize)
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
                self.drawJoint(joint.get_transform_matrix(), j)


